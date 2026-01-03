import argparse
import io
import json
import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    TrainingArguments, 
    Trainer,
    set_seed,
    BitsAndBytesConfig  # Para QLoRA
)
from peft import LoraConfig, get_peft_model
from src.common.minio_client import get_minio_client
import src.common.global_variables as config

# ---------------------------------------------------------
# 0. CONFIGURACIÓN DE REPRODUCIBILIDAD
# ---------------------------------------------------------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class MinioCLIPDataset(Dataset):
    def __init__(self, bucket_name, split_prefix, processor):
        self.client = get_minio_client()
        self.bucket_name = bucket_name
        self.processor = processor
        matches_key = f"{split_prefix}matches.json"
        obj = self.client.get_object(bucket_name, matches_key)
        self.matches = json.loads(obj.read().decode("utf-8"))
        obj.close(); obj.release_conn()

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        item = self.matches[idx]
        image_path = item["image_path"]
        text = item["text"]
        try:
            img_obj = self.client.get_object(self.bucket_name, image_path)
            image = Image.open(io.BytesIO(img_obj.read())).convert("RGB")
            img_obj.close(); img_obj.release_conn()
        except Exception as e:
            return None
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding="max_length", truncation=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, None, None)

# ---------------------------------------------------------
# 2. FUNCIÓN DE PLOT ACTUALIZADA (Dinámica para 1600 imágenes)
# ---------------------------------------------------------
def plot_training_results(trainer, rank, lr, method):
    history = trainer.state.log_history
    
    train_loss = [x["loss"] for x in history if "loss" in x]
    train_steps = [x["step"] for x in history if "loss" in x]
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    val_steps = [x["step"] for x in history if "eval_loss" in x]
    
    plt.figure(figsize=(12, 6))
    
    # 1600 imágenes / batch 16 = 100 pasos por época
    steps_per_epoch = 100 
    for epoch in range(1, 8):
        plt.axvline(x=epoch * steps_per_epoch, color='red', linestyle='--', alpha=0.3, label='Epoch End' if epoch == 1 else "")

    plt.plot(train_steps, train_loss, label=f"Train Loss ({method.upper()} r={rank})", color="#1f77b4", linewidth=2, alpha=0.6)
    
    if val_loss:
        plt.plot(val_steps, val_loss, label=f"Val Loss ({method.upper()} r={rank})", color="#ff7f0e", marker='o', linestyle='--', linewidth=2)
        for i, v in enumerate(val_loss):
            plt.text(val_steps[i], val_loss[i], f'{v:.4f}', color="#ff7f0e", fontweight='bold', ha='center', va='bottom')

    plt.xlabel("Steps")
    plt.ylabel("Loss Value")
    plt.title(f"Experiment CLIP {method.upper()}: Rank={rank}, LR={lr} (7 Epochs)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    filename = f"search_{method}_r{rank}_lr{lr}.png"
    save_path = os.path.join(config.EXPERIMENTS_DIR, filename)
    plt.savefig(save_path)
    plt.close() 
    print(f"Gráfica guardada en: {save_path}")


# 2. NUEVA FUNCIÓN: REPORTE DE EFICIENCIA
# ---------------------------------------------------------
def print_efficiency_report(method, rank, lr, trainable_params, all_params, train_result, peak_mem):
    total_time = train_result.metrics["train_runtime"]
    samples_per_second = train_result.metrics["train_samples_per_second"]
    
    print("\n" + "="*40)
    print(f"📊 REPORTE DE EFICIENCIA: {method.upper()}")
    print(f"Configuración: Rank={rank}, LR={lr}")
    print("-" * 40)
    print(f"✅ Parámetros Entrenables: {trainable_params:,}")
    print(f"✅ % del Modelo Original: {100 * trainable_params / all_params:.4f}%")
    print(f"✅ Tiempo Total: {total_time:.2f} segundos")
    print(f"✅ Velocidad: {samples_per_second:.2f} imágenes/seg")
    print(f"✅ Memoria VRAM Pico: {peak_mem:.2f} GB")
    print("="*40 + "\n")    

# ---------------------------------------------------------
# 3. FUNCIÓN DE ENTRENAMIENTO DINÁMICA
# ---------------------------------------------------------
def resolve_device(device):
    device = device.lower()
    if device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"
    elif device == "gpu" and not torch.cuda.is_available():
        device = "cpu"
    return device

def run_hyperparameter_experiment(rank, lr, method, device):
    device = resolve_device(device)
    print(f"\n" + "="*50)
    print(f"EJECUTANDO: {method.upper()} | Rank={rank}, LR={lr}, Device={device}")
    print("="*50)

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    
    # Configuración de cuantización solo si es QLoRA
    bnb_config = None
    if method == "qlora" and device == "gpu":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    model = CLIPModel.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto" if bnb_config else None
    )
    
    if device == "gpu" and not bnb_config:
        model.to("cuda")

    model.config.return_loss = True 

    lora_config = LoraConfig(
        r=rank, 
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # [MÉTRICA 1]: CAPTURAR PARÁMETROS ENTRENABLES
    trainable_params, all_params = model.get_nb_trainable_parameters()

    train_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TRAIN, processor)
    dev_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_DEV, processor)

    training_args = TrainingArguments(
        output_dir="./temp", 
        use_cpu=(device == "cpu"),
        per_device_train_batch_size=16,
        num_train_epochs=7,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no", 
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds
    )

    # [MÉTRICA 2]: REINICIAR ESTADÍSTICAS DE MEMORIA VRAM ANTES DE EMPEZAR
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # [MÉTRICA 3]: INICIAR ENTRENAMIENTO Y CAPTURAR TIEMPO
    train_result = trainer.train()
    
    # [MÉTRICA 4]: CALCULAR MEMORIA PICO AL TERMINAR
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) # Convertir a GB

    # LLAMADA A TU NUEVA FUNCIÓN DE REPORTE
    # Asegúrate de haber definido 'print_efficiency_report' antes
    print_efficiency_report(
        method, 
        rank, 
        lr, 
        trainable_params, 
        all_params, 
        train_result, 
        peak_mem
    )

    plot_training_results(trainer, rank, lr, method)
    
    # Limpiar memoria para el siguiente experimento
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# ---------------------------------------------------------
# 4. BUCLE PRINCIPAL (Búsqueda simétrica)
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    # ESPACIO DE BÚSQUEDA: 3 para LORA y las mismas 3 para QLORA
    search_space = [
        {"method": "lora",  "rank": 16, "lr": 5e-5},
        {"method": "lora",  "rank": 32, "lr": 5e-5},
        {"method": "lora",  "rank": 16, "lr": 1e-4},
        #{"method": "qlora", "rank": 16, "lr": 5e-5},
        #{"method": "qlora", "rank": 32, "lr": 5e-5},
        #{"method": "qlora", "rank": 16, "lr": 1e-4},
    ]

    for exp in search_space:
        run_hyperparameter_experiment(
            rank=exp["rank"], 
            lr=exp["lr"], 
            method=exp["method"],
            device=args.device
        )
    
    print("\n>>> BÚSQUEDA DE HIPERPARAMETROS LORA/QLORA COMPLETADA.")
