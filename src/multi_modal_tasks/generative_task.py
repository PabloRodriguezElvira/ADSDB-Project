import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    CLIPModel,
    CLIPProcessor
)
import os
from sentence_transformers import SentenceTransformer
from src.common.chroma_client import get_client, get_text_collection, get_image_collection

# Connection to ChromaDB
os.environ["HF_HUB_OFFLINE"] = "1"  # ✅ modo sin conexión
client = get_client()
print(client)
col_text = get_text_collection(client)
col_img  = get_image_collection(client)

text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text_encoder, text: str):# embeddings from the input text
    emb = text_encoder.encode(text, normalize_embeddings=True).tolist()
    return emb

def get_image_embedding(clip_model, clip_proc, image_path: str):# embeddings from the input image
    # get the image
    img = Image.open(image_path).convert("RGB")

    # preprocess (resize, normalize, tensor) in order to pass it to get tnhe embeddings
    processed = clip_proc(images=img, return_tensors="pt")
    pixel_values = processed["pixel_values"]  # <- más claro

    # get the embeddings
    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=pixel_values)
        normalized = torch.nn.functional.normalize(features, dim=-1)

    #  converting to list available with chroma
    embedding = normalized.squeeze().cpu().numpy().tolist()

    return embedding, img

def retrieve_from_chroma(col_text, col_img, text_emb, image_emb, k_text=3, k_img=2):
    # get the most similar text from chroma
    res_text = col_text.query(
        query_embeddings=[text_emb],
        n_results=k_text,
        include=["documents", "metadatas"]
    )
    # get the most similar images from chroma
    res_img = col_img.query(
        query_embeddings=[image_emb],
        n_results=k_img,
        include=["metadatas"]
    )

    return res_text, res_img

def build_prompt(user_query, res_text, res_img):
    system_prompt = (
        "You are a professional chief and you wanna make the best cooking advices. "
        "Use the recipes and images found to create the best answers. "
        "Explain the plate with the references found."
    )

    # Recipes found
    recipes = ""
    for doc, meta in zip(res_text["documents"][0], res_text["metadatas"][0]):
      # First 20 letters as title
      title = doc[:20]

      # The rest of words as ingredients and directions
      content = doc[20:]

      # Add a correct prompt
      recipes += f"- {title}:\n  {content[:250]}...\n\n"

    # Images found
    images = ""
    image_paths = []
    for meta in res_img["metadatas"][0]:
        path = meta.get("source_key", "")
        if path:
            images += f"- {path}\n"
            image_paths.append(path)

    user_prompt = (
        f"User is asking for: '{user_query}'.\n\n"
        f"Here you got some related recipes:\n{recipes}\n"
        f"And some related images:\n{images}\n"
        f"Suggest the best possible recipe according to what the user asked."
    )

    return system_prompt, user_prompt, image_paths


# ============================================================
# 5️⃣ Generación de respuesta (LLaVA)
# ============================================================

def load_llava_model():
    """Load a public LLaVA model from Hugging Face"""
    MODEL_ID =  "llava-hf/llava-1.5-7b-hf"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n Loading LLaVA model on {device}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("✅ LLaVA model loaded.\n")

    return model, processor, device


def generate_response(model, processor, device, query_image, system_prompt, user_prompt):
    """Genera una respuesta textual a partir del prompt y la imagen"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=query_image, text=chat_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=400, temperature=0.2)

    response = processor.decode(output[0], skip_special_tokens=True)
    return response


# ============================================================
# 6️⃣ Pipeline completo
# ============================================================


def rag_pipeline(user_query: str, image_path: str):
    """Runs the full multimodal RAG pipeline"""

    # Step 1: Get embeddings
    text_emb = get_text_embedding(text_encoder, user_query)
    image_emb, query_image = get_image_embedding(clip_model, clip_proc, image_path)

    # Step 2: Retrieve relevant info from Chroma
    res_text, res_img = retrieve_from_chroma(col_text, col_img, text_emb, image_emb)

    # Step 3: Build prompt
    system_prompt, user_prompt, image_paths = build_prompt(user_query, res_text, res_img)

    # Step 4: Load LLaVA model
    model, processor, device = load_llava_model()

    # Step 5: Generate final response
    response = generate_response(model, processor, device, query_image, system_prompt, user_prompt)

    return response, image_paths



# ============================================================
# 7️⃣ Ejecución del script
# ============================================================

if __name__ == "__main__":
    query = "Give me a recipe that contains pulled pork"
    image_path = r"C:\Users\adals\OneDrive\Documentos\Master\ADSDB-Project\pulled_pork.png"

    print("\n🚀 Ejecutando pipeline RAG...\n")

    # 🔹 Captura ambos valores que devuelve el pipeline:
    answer, image_paths = rag_pipeline(query, image_path)

    print(" FINAL RESPONSE FROM LLAVA:")
    print(answer)

    print("\n IMAGES FOUND:")
    for p in image_paths:
        print(f"- {p}")