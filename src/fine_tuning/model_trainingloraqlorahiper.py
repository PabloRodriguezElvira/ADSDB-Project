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

# Fix a SEED to obtain the same results
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class MinioCLIPDataset(Dataset):
    """
    Custom Dataset in order to sream images and text directly from MinIO bucket.
    This helps to avoid local storage bottlenecks during large-scale training.
    """
    def __init__(self, bucket_name, split_prefix, processor):
        self.client = get_minio_client()
        self.bucket_name = bucket_name
        self.processor = processor
        # Load the JSON that links image-text pairs.
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
            # Efficiently fetch the image from MinIO and convert to RGB.
            img_obj = self.client.get_object(self.bucket_name, image_path)
            image = Image.open(io.BytesIO(img_obj.read())).convert("RGB")
            img_obj.close(); img_obj.release_conn()
        except Exception as e:
            return None
        # Process image and text into tensors for CLIP (padding and truncation enabled).
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding="max_length", truncation=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}

class CLIPTrainer(Trainer):
    """
    This Custom Trainer is used to calculate the Loss and manage the training
    process for CLIP.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Loss computation
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Prediction step to focus on contrastive loss evaluation.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, None, None)


def plot_training_results(trainer, rank, lr, method):
    """
    Generates a chart showing how the loss decreases during training (LoRA and QLoRA).
    It marks each epoch to visualize the model's progress over time.
    """
    history = trainer.state.log_history
    # Extract training and validation loss values from the logs
    train_loss = [x["loss"] for x in history if "loss" in x]
    train_steps = [x["step"] for x in history if "loss" in x]
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    val_steps = [x["step"] for x in history if "eval_loss" in x]
    
    plt.figure(figsize=(12, 6))
    
    # we use 100 steps_per_epoch because 1600img / batch 16 = 100 steps per epoch
    # Draw vertical lines to mark the end of each epoch (100 steps per epoch)
    steps_per_epoch = 100 
    for epoch in range(1, 8):
        plt.axvline(x=epoch * steps_per_epoch, color='red', linestyle='--', alpha=0.3, label='Epoch End' if epoch == 1 else "")
    # Plot the training loss curve
    plt.plot(train_steps, train_loss, label=f"Train Loss ({method.upper()} r={rank})", color="#1f77b4", linewidth=2, alpha=0.6)
    # Plot validation loss points and show the exact value for each one
    if val_loss:
        plt.plot(val_steps, val_loss, label=f"Val Loss ({method.upper()} r={rank})", color="#ff7f0e", marker='o', linestyle='--', linewidth=2)
        for i, v in enumerate(val_loss):
            plt.text(val_steps[i], val_loss[i], f'{v:.4f}', color="#ff7f0e", fontweight='bold', ha='center', va='bottom')

    plt.xlabel("Steps")
    plt.ylabel("Loss Value")
    plt.title(f"Experiment CLIP {method.upper()}: Rank={rank}, LR={lr} (7 Epochs)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    filename = f"search_{method}_r{rank}_lr{lr}.png" # saving PNG files in the experiments folder
    save_path = os.path.join(config.EXPERIMENTS_DIR, filename)
    plt.savefig(save_path)
    plt.close() 
    print(f"Image saved in: {save_path}")


def print_efficiency_report(method, rank, lr, trainable_params, all_params, train_result, peak_mem):
    """
    Calculates and prints the main efficiency results. 
    It tracks how much memory and time the model uses.
    """
    # Extract timing and speed data from the training results
    total_time = train_result.metrics["train_runtime"]
    samples_per_second = train_result.metrics["train_samples_per_second"]
    
    print("\n" + "="*40)
    print(f"Efficiency Metrics: {method.upper()}")
    print(f"Configuration: Rank={rank}, LR={lr}")
    print("-" * 40)
    print(f"Trainable Parametres: {trainable_params:,}")# Amount of trainable parameters
    print(f"% of Trainable Parameters: {100 * trainable_params / all_params:.4f}%")# % Trainable Parameters
    print(f"Total Time: {total_time:.2f} seconds")# Total time
    print(f"Speed: {samples_per_second:.2f} img/sec")# Speed img/sec
    print(f"VRAM : {peak_mem:.2f} GB")# Peak VRAM usage
    print("="*40 + "\n")    

# ---------------------------------------------------------
# 3. FUNCIÓN DE ENTRENAMIENTO DINÁMICA
# ---------------------------------------------------------
def resolve_device(device):
    """Checks if a GPU is available; otherwise, defaults to CPU."""
    device = device.lower()
    if device == "gpu" and not torch.cuda.is_available():
        device = "cpu"
    return device

def run_hyperparameter_experiment(rank, lr, method, device):
    """
    Main function to run a single training experiment with specific settings.
    It handles model loading, quantization, LoRA setup, and training.
    """
    device = resolve_device(device)
    print(f"\n" + "="*50)
    print(f"Running: {method.upper()} | Rank={rank}, LR={lr}, Device={device}")
    print("="*50)

    model_id = config.MODEL_CLIP# assign our CLIP model chosen
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
    
    # Only for QLoRA on GPU)
    bnb_config = None
    if method == "qlora" and device == "gpu":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    # Load the base CLIP model with optional quantization
    model = CLIPModel.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto" if bnb_config else None
    )
    
    if device == "gpu" and not bnb_config:
        model.to("cuda")

    model.config.return_loss = True # Returns Loss
    # LoRA Configuration: Defines how we adapt the model
    lora_config = LoraConfig(
        r=rank, # Rank takes the different values to compare them
        lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    # Apply Parameter-Efficient Fine-Tuning (PEFT)
    model = get_peft_model(model, lora_config)

    # Metric 1: Calculate the fraction of the model being trained
    trainable_params, all_params = model.get_nb_trainable_parameters()

    # Initialize the datasets from MinIO
    train_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TRAIN, processor)
    dev_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_DEV, processor)
    # Define training parameters (batch size, epochs, etc.)
    training_args = TrainingArguments(
        #output_dir="./temp", # no output because in testing we will train it with optimal parameters
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

    # Metric 2: Reset VRAM tracking before training starts
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Metric 3: Start training and record the duration
    train_result = trainer.train()
    
    # Metric 4: Capture peak VRAM usage in GB
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) # Convertir a GB

    # Generate efficiency report and training plots
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
    
    # Memory Cleanup: Prepare for the next experiment
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # To select between CPU or GPU execution
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()

    # Defining different configurations for LoRA and QLoRA
    search_space = [
        {"method": "lora",  "rank": 16, "lr": 5e-5},
        {"method": "lora",  "rank": 32, "lr": 5e-5},
        {"method": "lora",  "rank": 16, "lr": 1e-4},
        {"method": "qlora", "rank": 16, "lr": 5e-5},
        {"method": "qlora", "rank": 32, "lr": 5e-5},
        {"method": "qlora", "rank": 16, "lr": 1e-4}
    ]
    # Automated execution loop: runs each experiment one by one
    for exp in search_space:
        run_hyperparameter_experiment(
            rank=exp["rank"], 
            lr=exp["lr"], 
            method=exp["method"],
            device=args.device
        )
    
    print("\n>>> Hyperparameter search completed for LoRA/QLoRA.")