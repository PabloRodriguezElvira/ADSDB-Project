import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import random
from PIL import Image
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    TrainingArguments, 
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model
from sklearn.decomposition import PCA
import re

# IMPORTACIÓN DE TUS CLASES
from src.fine_tuning.model_trainingloraqlorahiper import MinioCLIPDataset, CLIPTrainer
import src.common.global_variables as config

# 0. CONFIGURACIÓN
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = config.MODEL_CLIP
processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)

# --- NUEVA FUNCIÓN: DESCARGA Y SIMILITUD ---
def download_top_5_with_scores(model_zs, model_qlora, dataset, sample_idx=None):
    """
    Busca, imprime, calcula similitudes y DESCARGA las imágenes en carpetas locales.
    """
    if sample_idx is None:
        #sample_idx = random.randint(0, len(dataset) - 1)
        sample_idx = 2
    
    query_text = dataset.matches[sample_idx]["text"]
    correct_path = dataset.matches[sample_idx]["image_path"]
    
    # Crear carpeta específica para esta muestra en image_experiments
    base_folder = os.path.join(config.EXPERIMENTS_DIR, f"retrieval_sample_{sample_idx}")
    os.makedirs(os.path.join(base_folder, "zero_shot"), exist_ok=True)
    os.makedirs(os.path.join(base_folder, "qlora"), exist_ok=True)

    model_zs.eval()
    model_qlora.eval()

    # 1. Embeddings de texto normalizados
    max_length = getattr(model_zs.config.text_config, "max_position_embeddings", None)
    if max_length is None:
        max_length = getattr(model_zs.config, "max_position_embeddings", 77)
    text_inputs = processor(
        text=[query_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        text_emb_zs = model_zs.get_text_features(**text_inputs)
        text_emb_ql = model_qlora.get_text_features(**text_inputs)
        text_emb_zs /= text_emb_zs.norm(dim=-1, keepdim=True)
        text_emb_ql /= text_emb_ql.norm(dim=-1, keepdim=True)
    
    # 2. Embeddings de todas las imágenes del test
    img_embs_zs, img_embs_ql, paths = [], [], []
    print(f"Procesando búsqueda para receta idx {sample_idx}...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset.matches[i]
            paths.append(item["image_path"])
            
            # Obtener imagen de MinIO
            obj = dataset.client.get_object(dataset.bucket_name, item["image_path"])
            img_data = obj.read()
            img_raw = Image.open(io.BytesIO(img_data)).convert("RGB")
            obj.close(); obj.release_conn()
            
            inputs = processor(images=img_raw, return_tensors="pt").to(device)
            f_zs = model_zs.get_image_features(**inputs)
            f_ql = model_qlora.get_image_features(**inputs)
            
            img_embs_zs.append(f_zs / f_zs.norm(dim=-1, keepdim=True))
            img_embs_ql.append(f_ql / f_ql.norm(dim=-1, keepdim=True))

    # 3. Calcular similitudes de coseno
    scores_zs = (text_emb_zs @ torch.cat(img_embs_zs).t()).squeeze(0)
    scores_ql = (text_emb_ql @ torch.cat(img_embs_ql).t()).squeeze(0)
    
    top5_zs = scores_zs.argsort(descending=True)[:5]
    top5_ql = scores_ql.argsort(descending=True)[:5]

    # 4. Guardar imágenes y generar reporte .txt
    def save_results(indices, scores, model_name):
        report = [f"REPORT FOR {model_name.upper()}\n", f"Query: {query_text}\n\n"]
        for rank, idx in enumerate(indices, 1):
            idx = idx.item()
            sim = scores[idx].item()
            path = paths[idx]
            match = "CORRECT" if path == correct_path else "wrong"
            
            # Guardar archivo localmente
            obj = dataset.client.get_object(dataset.bucket_name, path)
            data = obj.read()
            obj.close(); obj.release_conn()
            
            filename = f"rank{rank}_sim{sim:.4f}_{match}.jpg"
            with open(os.path.join(base_folder, model_name, filename), "wb") as f:
                f.write(data)
            
            report.append(f"Rank {rank}: Sim={sim:.4f} | {path} | {'✅' if match=='CORRECT' else '❌'}\n")
        return report

    report_zs = save_results(top5_zs, scores_zs, "zero_shot")
    report_ql = save_results(top5_ql, scores_ql, "qlora")

    with open(os.path.join(base_folder, "similarity_report.txt"), "w", encoding="utf-8") as f:
        f.writelines(report_zs + ["\n" + "-"*30 + "\n"] + report_ql)

    print(f"\n>>> Éxito. Resultados guardados en: {base_folder}")

# --- TUS FUNCIONES DE MÉTRICAS (Loss y Barras) ---

def get_top_k_stats(model, dataloader, device):
    model.eval()
    all_image_embeds, all_text_embeds = [], []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            all_image_embeds.append(img_emb); all_text_embeds.append(txt_emb)
    image_features = torch.cat(all_image_embeds)
    text_features = torch.cat(all_text_embeds)
    logits = text_features @ image_features.t()
    _, top_indices = logits.topk(5, dim=-1)
    ground_truth = torch.arange(len(text_features)).to(device).view(-1, 1)
    hits = (top_indices == ground_truth)
    results = [hits[:, :k].any(dim=1).sum().item() for k in range(1, 6)]
    results.append(len(text_features) - results[-1])
    return results

def plot_top_k_retrieval(stats_zs, stats_qlora):
    labels = ['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5', 'Out of 5']
    x = np.arange(len(labels))
    width = 0.4 
    plt.figure(figsize=(12, 7))
    bar1 = plt.bar(x - width/2, stats_zs, width, label='Zero-Shot', color='#94a3b8', edgecolor='white')
    bar2 = plt.bar(x + width/2, stats_qlora, width, label='QLoRA Champion', color='#ff7f0e', edgecolor='white')
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    plt.title('Top-5 Image Retrieval (Test Set)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of Images (Total: 200)')
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig(os.path.join(config.EXPERIMENTS_DIR, "top_k_retrieval_bars.png"))
    plt.show()

def plot_final_test_evolution(history, loss_zs_baseline):
    test_losses = [x["eval_loss"] for x in history if "eval_loss" in x]
    epochs = [i + 1 for i in range(len(test_losses))]
    plt.figure(figsize=(10, 6))
    plt.axhline(y=loss_zs_baseline, color='black', linestyle='-.', linewidth=2, label=f"Zero-Shot Baseline ({loss_zs_baseline:.4f})")
    plt.text(0.6, loss_zs_baseline + 0.005, f'Baseline: {loss_zs_baseline:.4f}', color="black", fontweight='bold')
    plt.plot(epochs, test_losses, label="QLoRA Performance (Test Set)", color="#ff7f0e", marker='o', linewidth=2)
    for i, v in enumerate(test_losses):
        plt.text(epochs[i], v + 0.005, f'{v:.4f}', color="#ff7f0e", fontweight='bold', ha='center')
    plt.xlabel("Epochs"); plt.ylabel("Loss Value (Test Set)"); plt.xticks(epochs)
    plt.legend(loc='upper right'); plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(config.EXPERIMENTS_DIR, "final_test_evolution.png"))
    plt.show()

def generate_pca(model, model_name, dataset, device, filename):
    """
    Genera un gráfico PCA individual para un modelo específico.
    """
    model.eval()
    img_features, txt_features, ids = [], [], []
    
    print(f"\nExtrayendo embeddings para PCA de {model_name}...")
    with torch.no_grad():
        # Procesamos los primeros 50 para mantener la legibilidad
        for i in range(min(50, len(dataset))):
            item = dataset.matches[i]
            img_id = re.findall(r'\d+', item["image_path"])[-1]
            ids.append(img_id)

            # Imagen
            obj = dataset.client.get_object(dataset.bucket_name, item["image_path"])
            img_raw = Image.open(io.BytesIO(obj.read())).convert("RGB")
            obj.close(); obj.release_conn()
            inputs_img = processor(images=img_raw, return_tensors="pt").to(device)
            f_img = model.get_image_features(**inputs_img)
            img_features.append(f_img / f_img.norm(dim=-1, keepdim=True))

            # Texto
            inputs_txt = processor(text=[item["text"]], return_tensors="pt", 
                                   padding=True, truncation=True, max_length=77).to(device)
            f_txt = model.get_text_features(**inputs_txt)
            txt_features.append(f_txt / f_txt.norm(dim=-1, keepdim=True))

    # Preparar datos para PCA
    X_img = torch.cat(img_features).cpu().numpy()
    X_txt = torch.cat(txt_features).cpu().numpy()
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.vstack([X_img, X_txt]))
    c_img, c_txt = coords[:len(X_img)], coords[len(X_img):]

    # Configuración del gráfico
    plt.figure(figsize=(12, 9))
    plt.scatter(c_img[:, 0], c_img[:, 1], c='#1f77b4', marker='s', s=100, alpha=0.6, label='Images')
    plt.scatter(c_txt[:, 0], c_txt[:, 1], c='#ff7f0e', marker='o', s=100, alpha=0.6, label='Texts')

    # Dibujar líneas de unión y etiquetas para los primeros 20 pares
    for i in range(min(20, len(ids))):
        plt.plot([c_img[i, 0], c_txt[i, 0]], [c_img[i, 1], c_txt[i, 1]], 'gray', linestyle='--', alpha=0.3)
        plt.annotate(ids[i], (c_img[i, 0], c_img[i, 1]), fontsize=9, fontweight='bold', color='#1f77b4')
        plt.annotate(ids[i], (c_txt[i, 0], c_txt[i, 1]), fontsize=9, fontweight='bold', color='#ff7f0e')

    plt.title(f"PCA (Embedding space): {model_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    save_path = os.path.join(config.EXPERIMENTS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f">>> PCA {model_name} guardado en: {save_path}")


if __name__ == "__main__":
    # 1. BASELINE ZERO-SHOT
    print("Evaluando Zero-Shot...")
    model_zs = CLIPModel.from_pretrained(model_id).to(device)
    test_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TEST, processor)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)
    
    loss_zs = 0
    model_zs.eval()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            loss_zs += model_zs(**inputs, return_loss=True).loss.item()
    loss_zs /= len(test_loader)
    
    stats_zs = get_top_k_stats(model_zs, test_loader, device)

    # 2. ENTRENAMIENTO QLORA CHAMPION
    print("\nEntrenando QLoRA Champion...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model_qlora = CLIPModel.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    lora_rank = 16
    lora_lr = 1e-4
    model_qlora = get_peft_model(
        model_qlora,
        LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, target_modules=["q_proj", "v_proj"])
    )
   

    train_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TRAIN, processor)
    training_args = TrainingArguments(
        output_dir="./final_test_run", per_device_train_batch_size=16, 
        num_train_epochs=5, learning_rate=lora_lr, eval_strategy="epoch", 
        report_to="none", remove_unused_columns=False
    )
    trainer = CLIPTrainer(model=model_qlora, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)


    # 3. RESULTADOS FINALES Y DESCARGA CUALITATIVA
    stats_qlora = get_top_k_stats(model_qlora, test_loader, device)
    
    # Gráficas
    plot_final_test_evolution(trainer.state.log_history, loss_zs)
    plot_top_k_retrieval(stats_zs, stats_qlora)
    
    # Descarga de ejemplo cualitativo (Punto 5 del reporte)
    # Puedes cambiar sample_idx por el número de la receta que quieras investigar
    download_top_5_with_scores(model_zs, model_qlora, test_ds, sample_idx=None)
    generate_pca(model_zs, "Zero-Shot Baseline", test_ds, device, "pca_zero_shot.png")
    # 2. Generar para QLoRA
    generate_pca(model_qlora, "QLoRA Champion", test_ds, device, "pca_qlora.png")
