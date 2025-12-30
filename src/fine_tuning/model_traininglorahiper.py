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
    set_seed
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

# [Dataset y CLIPTrainer se mantienen igual que en tu código original]
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
        # Force eval loss computation even without labels.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, None, None)

# ---------------------------------------------------------
# 2. FUNCIÓN DE PLOT ACTUALIZADA (Acepta Parámetros)
# ---------------------------------------------------------
def plot_training_results(trainer, rank, lr):
    history = trainer.state.log_history
    
    train_loss = [x["loss"] for x in history if "loss" in x]
    train_steps = [x["step"] for x in history if "loss" in x]
    
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    val_steps = [x["step"] for x in history if "eval_loss" in x]
    
    plt.figure(figsize=(12, 6))
    
    # --- LÍNEAS VERTICALES POR ÉPOCA ---
    # Como tienes 7 épocas y cada una son 50 pasos:
    steps_per_epoch = 50 
    for epoch in range(1, 8): # De 1 a 7
        plt.axvline(x=epoch * steps_per_epoch, color='red', linestyle='--', alpha=0.3, label='Epoch End' if epoch == 1 else "")

    # Training Line
    plt.plot(train_steps, train_loss, label=f"Train Loss (r={rank})", color="#1f77b4", linewidth=2, alpha=0.6)
    
    # Validation Points
    if val_loss:
        plt.plot(val_steps, val_loss, label=f"Val Loss (r={rank})", color="#ff7f0e", marker='o', linestyle='--', linewidth=2)
        for i, v in enumerate(val_loss):
            plt.text(val_steps[i], val_loss[i], f'{v:.4f}', color="#ff7f0e", fontweight='bold', ha='center', va='bottom')

    plt.xlabel("Steps")
    plt.ylabel("Loss Value")
    plt.title(f"Experiment CLIP LORA: Rank={rank}, LR={lr} (7 Epochs)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # Guardado
    filename = f"search_results_r{rank}_lr{lr}.png"
    save_path = os.path.join(config.EXPERIMENTS_DIR, filename)
    plt.savefig(save_path)
    plt.close() 
    
    print(f"Gráfica guardada exitosamente en: {save_path}")

# ---------------------------------------------------------
# 3. FUNCIÓN DE ENTRENAMIENTO DINÁMICA
# ---------------------------------------------------------
def resolve_device(device):
    device = device.lower()
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if device == "gpu" and not torch.cuda.is_available():
        print("GPU solicitada pero no disponible. Usando CPU.")
        device = "cpu"
    return device


def run_hyperparameter_experiment(rank, lr, device="cpu"):
    device = resolve_device(device)
    print(f"\n" + "="*50)
    print(f"EJECUTANDO: Rank={rank}, Learning Rate={lr}, Device={device}")
    print("="*50)

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    model = CLIPModel.from_pretrained(model_id)
    model.config.return_loss = True 

    # Aplicar LoRA con el Rank actual del bucle
    lora_config = LoraConfig(
        r=rank, 
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    train_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TRAIN, processor)
    dev_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_DEV, processor)

    training_args = TrainingArguments(
        #output_dir=f"./output_r{rank}_lr{lr}", # Carpeta única
        use_cpu=(device == "cpu"),
        per_device_train_batch_size=16,# en training 16 y le añado en validation el defecto q es 8
        num_train_epochs=7,
        learning_rate=lr,
        eval_strategy="epoch",# hacer el proceso de validaciion cada vez que termine una epoca del training 
        save_strategy="epoch",
        logging_steps=5,
        #load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds
    )

    trainer.train()
    plot_training_results(trainer, rank, lr)

# ---------------------------------------------------------
# 4. BUCLE PRINCIPAL DE BÚSQUEDA
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busqueda de hiperparametros para CLIP con LoRA.")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Dispositivo de entrenamiento (cpu o gpu).",
    )
    args = parser.parse_args()
    # Definimos el espacio de búsqueda (tus 3 combinaciones)
    search_space = [
        {"rank": 16, "lr": 5e-5}, # Baseline
        {"rank": 32, "lr": 5e-5}, # Mayor capacidad 
        {"rank": 16, "lr": 1e-4}  # Mayor velocidad 
    ]

    for experiment in search_space:
        run_hyperparameter_experiment(
            rank=experiment["rank"], # este se añade en lora pq tiene que ver con el cuerpo del modelo (capacidad)
            lr=experiment["lr"], # este va en training arguments pq es un optimizador que solo cotnrola la velocidad
            device=args.device,
        )
    
    print("\n>>> BÚSQUEDA DE HIPERPARAMETROS COMPLETADA.")
