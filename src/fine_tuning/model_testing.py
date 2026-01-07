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
    set_seed,
)
from peft import LoraConfig, get_peft_model
from sklearn.decomposition import PCA
import re

from src.fine_tuning.model_trainingloraqlorahiper import MinioCLIPDataset, CLIPTrainer
import src.common.global_variables as config

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = config.MODEL_CLIP# assign our CLIP model chose
processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)


def download_top_5_with_scores(model_zs, model_qlora, dataset, sample_idx=None):
    """Run retrieval for one sample and save the top-5 images with scores."""
    if sample_idx is None:
        sample_idx = 2

    query_text = dataset.matches[sample_idx]["text"]
    correct_path = dataset.matches[sample_idx]["image_path"]

    base_folder = os.path.join(config.EXPERIMENTS_DIR, f"retrieval_sample_{sample_idx}")
    os.makedirs(os.path.join(base_folder, "zero_shot"), exist_ok=True)
    os.makedirs(os.path.join(base_folder, "qlora"), exist_ok=True)

    model_zs.eval()
    model_qlora.eval()

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
    # Normalize embeddings to calculate Cosine Similarity later
    with torch.no_grad():
        text_emb_zs = model_zs.get_text_features(**text_inputs)
        text_emb_ql = model_qlora.get_text_features(**text_inputs)
        text_emb_zs /= text_emb_zs.norm(dim=-1, keepdim=True)
        text_emb_ql /= text_emb_ql.norm(dim=-1, keepdim=True)

    img_embs_zs, img_embs_ql, paths = [], [], []
    print(f"Processing recipe search {sample_idx}...")

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset.matches[i]
            paths.append(item["image_path"])

            obj = dataset.client.get_object(dataset.bucket_name, item["image_path"])
            img_data = obj.read()
            img_raw = Image.open(io.BytesIO(img_data)).convert("RGB")
            obj.close()
            obj.release_conn()

            inputs = processor(images=img_raw, return_tensors="pt").to(device)
            f_zs = model_zs.get_image_features(**inputs)
            f_ql = model_qlora.get_image_features(**inputs)

            img_embs_zs.append(f_zs / f_zs.norm(dim=-1, keepdim=True))
            img_embs_ql.append(f_ql / f_ql.norm(dim=-1, keepdim=True))
    # Matrix multiplication to get similarity scores between the text and all images
    scores_zs = (text_emb_zs @ torch.cat(img_embs_zs).t()).squeeze(0)
    scores_ql = (text_emb_ql @ torch.cat(img_embs_ql).t()).squeeze(0)
    # Get the indexes of the 5 images with the highest similarity scores
    top5_zs = scores_zs.argsort(descending=True)[:5]
    top5_ql = scores_ql.argsort(descending=True)[:5]

    def save_results(indices, scores, model_name):
        """Save ranked images and return report lines."""
        report = [f"REPORT FOR {model_name.upper()}\n", f"Query: {query_text}\n\n"]
        for rank, idx in enumerate(indices, 1):
            idx = idx.item()
            sim = scores[idx].item()
            path = paths[idx]
            match = "CORRECT" if path == correct_path else "wrong"

            obj = dataset.client.get_object(dataset.bucket_name, path)
            data = obj.read()
            obj.close()
            obj.release_conn()

            filename = f"rank{rank}_sim{sim:.4f}_{match}.jpg"
            with open(os.path.join(base_folder, model_name, filename), "wb") as f:
                f.write(data)

            report.append(
                f"Rank {rank}: Sim={sim:.4f} | {path} | {'ƒo.' if match=='CORRECT' else 'ƒ?O'}\n"
            )
        return report

    report_zs = save_results(top5_zs, scores_zs, "zero_shot")
    report_ql = save_results(top5_ql, scores_ql, "qlora")

    with open(
        os.path.join(base_folder, "similarity_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines(report_zs + ["\n" + "-" * 30 + "\n"] + report_ql)

    print(f"\n>>> Saved in: {base_folder}")


def get_top_k_stats(model, dataloader, device):
    """
    Quantitative evaluation: Calculates Accuracy at Top-1, Top-2... up to Top-5.
    It measures how often the correct image appears in the first 'k' results.
    """
    model.eval()
    all_image_embeds, all_text_embeds = [], []
    # Generate embeddings for the entire test set
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            all_image_embeds.append(img_emb)
            all_text_embeds.append(txt_emb)
    image_features = torch.cat(all_image_embeds)
    text_features = torch.cat(all_text_embeds)
    # Compute the similarity matrix for all pairs
    logits = text_features @ image_features.t()
    # Check if the diagonal (correct pair) is within the Top-K indices
    _, top_indices = logits.topk(5, dim=-1)
    ground_truth = torch.arange(len(text_features)).to(device).view(-1, 1)
    hits = top_indices == ground_truth
    # Sum total hits for each k between 1 to 5
    results = [hits[:, :k].any(dim=1).sum().item() for k in range(1, 6)]
    results.append(len(text_features) - results[-1])
    return results


def plot_top_k_retrieval(stats_zs, stats_qlora):
    """
    Creates a bar chart comparing Zero-Shot vs. QLoRA performance.
    Visualizes the improvement in retrieval accuracy across the test set.
    """
    labels = ["Top-1", "Top-2", "Top-3", "Top-4", "Top-5", "Out of 5"]
    x = np.arange(len(labels))
    width = 0.4
    plt.figure(figsize=(12, 7))
    bar1 = plt.bar(
        x - width / 2, stats_zs, width, label="Zero-Shot", color="#94a3b8", edgecolor="white"
    )
    bar2 = plt.bar(
        x + width / 2,
        stats_qlora,
        width,
        label="QLoRA Champion",
        color="#ff7f0e",
        edgecolor="white",
    )
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.annotate(
            f"{int(height)}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.title("Top-5 Image Retrieval (Test Set)", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Number of Images (Total: 200)")
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig(os.path.join(config.EXPERIMENTS_DIR, "top_k_retrieval_bars.png"))# saved in the experiments folder
    plt.show()


def plot_final_test_evolution(history, loss_zs_baseline):
    """"
    Comparison between the best trained model and the original Zero-Shot performance.
    It plots the evaluation Test loss over epochs to show continuous improvement
    relative to the initial baseline.
    """
    # Extract evaluation loss values from training logs
    test_losses = [x["eval_loss"] for x in history if "eval_loss" in x]
    epochs = [i + 1 for i in range(len(test_losses))]
    plt.figure(figsize=(10, 6))
    # Zero-Shot Baseline: Shows the performance of the model WITHOUT training
    plt.axhline(
        y=loss_zs_baseline,
        color="black",
        linestyle="-.",
        linewidth=2,
        label=f"Zero-Shot Baseline ({loss_zs_baseline:.4f})",
    )
    plt.text(
        0.6,
        loss_zs_baseline + 0.005,
        f"Baseline: {loss_zs_baseline:.4f}",
        color="black",
        fontweight="bold",
    )
    # Plotting the QLoRA progress on the test set
    plt.plot(
        epochs,
        test_losses,
        label="QLoRA Performance (Test Set)",
        color="#ff7f0e",
        marker="o",
        linewidth=2,
    )
    for i, v in enumerate(test_losses):
        plt.text(
            epochs[i],
            v + 0.005,
            f"{v:.4f}",
            color="#ff7f0e",
            fontweight="bold",
            ha="center",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value (Test Set)")
    plt.xticks(epochs)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(config.EXPERIMENTS_DIR, "final_test_evolution.png"))# Saved in the experiments folder
    plt.show()


def generate_pca(model, model_name, dataset, device, filename):
    """
    PCA to visualize the embedding space.
    It reduces high-dimensional data (CLIP features) to 2D to see how 
    images and texts allign each other after training.
    """
    model.eval()
    img_features, txt_features, ids = [], [], []

    print(f"\nExtracting embeddings for PCA from {model_name}...")
    # Extract features for both modalities (Images and Texts)
    with torch.no_grad():
        for i in range(min(50, len(dataset))):
            item = dataset.matches[i]
            img_id = re.findall(r"\d+", item["image_path"])[-1]
            ids.append(img_id)

            obj = dataset.client.get_object(dataset.bucket_name, item["image_path"])
            img_raw = Image.open(io.BytesIO(obj.read())).convert("RGB")
            obj.close()
            obj.release_conn()
            inputs_img = processor(images=img_raw, return_tensors="pt").to(device)
            # Image Features
            f_img = model.get_image_features(**inputs_img)
            img_features.append(f_img / f_img.norm(dim=-1, keepdim=True))

            inputs_txt = processor(
                text=[item["text"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            # Text Features
            f_txt = model.get_text_features(**inputs_txt)
            txt_features.append(f_txt / f_txt.norm(dim=-1, keepdim=True))

    X_img = torch.cat(img_features).cpu().numpy()
    X_txt = torch.cat(txt_features).cpu().numpy()
    # PCA Projection: Merging image and text vectors into a single 2D space
    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.vstack([X_img, X_txt]))
    c_img, c_txt = coords[: len(X_img)], coords[len(X_img) :]

    plt.figure(figsize=(12, 9))
    plt.scatter(
        c_img[:, 0],
        c_img[:, 1],
        c="#1f77b4",
        marker="s",
        s=100,
        alpha=0.6,
        label="Images",
    )
    plt.scatter(
        c_txt[:, 0],
        c_txt[:, 1],
        c="#ff7f0e",
        marker="o",
        s=100,
        alpha=0.6,
        label="Texts",
    )
    # Draw lines connecting the same recipe (Image-Text)
    # Shorter lines mean better alignment between modalities
    for i in range(min(20, len(ids))):
        plt.plot(
            [c_img[i, 0], c_txt[i, 0]],
            [c_img[i, 1], c_txt[i, 1]],
            "gray",
            linestyle="--",
            alpha=0.3,
        )
        plt.annotate(
            ids[i],
            (c_img[i, 0], c_img[i, 1]),
            fontsize=9,
            fontweight="bold",
            color="#1f77b4",
        )
        plt.annotate(
            ids[i],
            (c_txt[i, 0], c_txt[i, 1]),
            fontsize=9,
            fontweight="bold",
            color="#ff7f0e",
        )

    plt.title(f"PCA (Embedding space): {model_name}", fontsize=16, fontweight="bold")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(config.EXPERIMENTS_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f">>> PCA {model_name} saved en: {save_path}")# Saved in experiments folder


if __name__ == "__main__":
    print("Evaluating Zero-Shot model")
    # We establish the initial performance of the model without any fine-tuning.
    model_zs = CLIPModel.from_pretrained(model_id).to(device)
    test_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TEST, processor)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16)

    loss_zs = 0
    model_zs.eval()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            # Measure the contrastive loss on the test set for comparison
            loss_zs += model_zs(**inputs, return_loss=True).loss.item()
    loss_zs /= len(test_loader)
    # Get retrieval statistics (Top-1 to Top-5) for the baseline
    stats_zs = get_top_k_stats(model_zs, test_loader, device)

    print("\Training QLoRA Champion...")
    # We use the best hyperparameters found during the training phase.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_qlora = CLIPModel.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    # Selected optimal configuration: Rank 16 and LR 1e-4
    lora_rank = 16
    lora_lr = 1e-4
    model_qlora = get_peft_model(
        model_qlora,
        LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2, target_modules=["q_proj", "v_proj"]),
    )

    train_ds = MinioCLIPDataset(config.TRAINING_DATASET_BUCKET, config.TRAINING_TRAIN, processor)
    training_args = TrainingArguments(
        #output_dir="./final_test_run", we do not need to save it
        per_device_train_batch_size=16,
        num_train_epochs=5,# Sufficient epochs to show convergence
        learning_rate=lora_lr,
        eval_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = CLIPTrainer(
        model=model_qlora, args=training_args, train_dataset=train_ds, eval_dataset=test_ds
    )
    # Track GPU memory usage during the champion training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    # Extract final statistics for the improved model
    stats_qlora = get_top_k_stats(model_qlora, test_loader, device)
    # Generate all comparison charts and reports
    plot_final_test_evolution(trainer.state.log_history, loss_zs)
    plot_top_k_retrieval(stats_zs, stats_qlora)
    # saving Top-5 images for a specific recipe
    download_top_5_with_scores(model_zs, model_qlora, test_ds, sample_idx=None)
    # Saving PCA in the experiments folder
    generate_pca(model_zs, "Zero-Shot Baseline", test_ds, device, "pca_zero_shot.png")
    generate_pca(model_qlora, "QLoRA Champion", test_ds, device, "pca_qlora.png")
