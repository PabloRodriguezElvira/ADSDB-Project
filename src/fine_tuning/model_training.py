import io
import json
import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    CLIPConfig,
    TrainingArguments, 
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model
from src.common.minio_client import get_minio_client
import src.common.global_variables as config

# ---------------------------------------------------------
# 0. CONFIGURACIÓN DE REPRODUCIBILIDAD (Constraint 9)
# ---------------------------------------------------------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------
# 1. CLASE DATASET (Constraint 8)
# ---------------------------------------------------------
class MinioCLIPDataset(Dataset):
    def __init__(self, bucket_name, split_prefix, processor):
        self.client = get_minio_client()
        self.bucket_name = bucket_name
        self.processor = processor
        
        # Cargar el JSON de matches desde MinIO
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
            # Descarga bajo demanda (Constraint 8)
            img_obj = self.client.get_object(self.bucket_name, image_path)
            image = Image.open(io.BytesIO(img_obj.read())).convert("RGB")
            img_obj.close(); img_obj.release_conn()
        except Exception as e:
            print(f"Error cargando imagen {image_path}: {e}")
            return None

        # Procesamiento para CLIP
        inputs = self.processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

# ---------------------------------------------------------
# 2. CUSTOM TRAINER PARA CLIP (Solución al ValueError)
# ---------------------------------------------------------
# CLIP no devuelve loss por defecto, así que forzamos su cálculo
class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forzamos return_loss=True en la llamada al modelo
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# ---------------------------------------------------------
# 3. FUNCIÓN DE ENTRENAMIENTO (LoRA - Constraint 1)
# ---------------------------------------------------------
def train_model():
    model_id = "openai/clip-vit-base-patch32"
    
    # A) CARGA EN CPU (Constraint 1: Eficiencia)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, device_map="cpu")
    
    # Forzamos configuración de pérdida
    model.config.return_loss = True 

    # B) PREPARAR LORA (Constraint 1)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # C) INSTANCIAR DATASETS
    train_ds = MinioCLIPDataset(
        config.TRAINING_DATASET_BUCKET, 
        config.TRAINING_TRAIN, 
        processor
    )
    dev_ds = MinioCLIPDataset(
        config.TRAINING_DATASET_BUCKET, 
        config.TRAINING_DEV, 
        processor
    )

    # D) ARGUMENTOS DE ENTRENAMIENTO
    training_args = TrainingArguments(
        #output_dir="./temp_checkpoints",
        use_cpu=True,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none"
    )

    # E) EJECUCIÓN CON CLIPTRAINER
    trainer = CLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds
    )

    print("Iniciando Fine-Tuning en CPU con Custom CLIP Loss...")
    trainer.train()

    # F) GUARDADO LOCAL (¡Descoméntalo cuando quieras guardar!)
    # model.save_pretrained("./final_adapter_local")
    print("Entrenamiento completado.")

if __name__ == "__main__":
    train_model()