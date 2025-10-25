import torch
import numpy as np
from PIL import Image
import minio
import os
import io
import google.generativeai as genai
import warnings
import src.common.global_variables as config  # 👈 para acceder a TRUSTED_BUCKET
from chromadb.utils import embedding_functions as ef
from minio.error import S3Error
from src.common.minio_client import get_minio_client
from diffusers import StableDiffusionPipeline
from src.common.chroma_client import (
    get_client,
    get_image_collection,
    get_text_collection,
    _text_ef,
    _image_ef,
)

# ============================================================
# 🔹 Configuración general y silenciamiento de warnings
# ============================================================

warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated",
    category=FutureWarning,
    module="huggingface_hub",
)
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_PLUGIN_LOGGER"] = "NONE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# 🔹 1️⃣ Conexión a ChromaDB
# ============================================================

client = get_client()
col_text = get_text_collection(client)
col_img = get_image_collection(client)

text_ef = ef.SentenceTransformerEmbeddingFunction(model_name=config.TEXT_MODEL_NAME)
image_ef = ef.OpenCLIPEmbeddingFunction()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY no está configurada. Usa una variable de entorno.")

genai.configure(api_key=api_key)

MODEL_ID =  "models/gemini-2.5-flash"     # Puedes cambiar a "models/gemini-2.5-pro" si lo deseas
model = genai.GenerativeModel(MODEL_ID)

#image_model = genai.GenerativeModel("models/gemini-2.5-flash-image")
# ============================================================
# 🔹 2️⃣ Funciones auxiliares
# ============================================================

def get_text_embedding(text: str):
    emb_list = text_ef([text])
    return emb_list[0]

def get_image_embedding(image_path: str):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    emb_list = image_ef([arr])
    return emb_list[0], img

def retrieve_from_chroma(col_text, col_img, text_emb, image_emb, k_text=2, k_img=2):
    res_text = col_text.query(
        query_embeddings=[text_emb],
        n_results=k_text,
        include=["documents", "metadatas"]
    )
    res_img = col_img.query(
        query_embeddings=[image_emb],
        n_results=k_img,
        include=["metadatas"]
    )
    return res_text, res_img



def get_images_from_minio_for_gemini(res_img):
    """Lee las imágenes recuperadas desde Chroma (en el bucket trusted-zone) y devuelve inline_data listo para Gemini."""
    client_s3 = get_minio_client()
    image_parts = []

    for meta in res_img.get("metadatas", [[]])[0]:
        source_key = meta.get("source_key")
        if not source_key:
            continue

        try:
            # ⚡ Usamos el bucket global de tu config, no del metadata
            bucket = config.TRUSTED_BUCKET  # = "trusted-zone"
            mime_type = "image/png"         # puedes mejorarlo si luego guardas esto en metadata

            # 📥 Descargar desde MinIO
            response = client_s3.get_object(bucket, source_key)
            data = response.read()
            response.close()
            response.release_conn()

            # (Opcional) validar que sea imagen
            Image.open(io.BytesIO(data)).verify()

            # ✅ Parte lista para Gemini
            image_parts.append({"inline_data": {"mime_type": mime_type, "data": data}})

        except Exception as e:
            print(f" Error descargando {source_key} desde MinIO ({bucket}): {e}")

    return image_parts

def build_prompt(user_query, res_text, res_img):
    system_prompt = (
        "You are a chef sharing cooking idea using the text and images found in a friendly tone." )
    recipes = ""
    for doc, meta in zip(res_text["documents"][0], res_text["metadatas"][0]):
        title = doc[:20]
        content = doc[20:]
        recipes += f"- {title}:\n  {content[:250]}...\n\n"

    images = ""
    image_paths = []
    for meta in res_img["metadatas"][0]:
        path = meta.get("source_key", "")
        if path:
            images += f"- {path}\n"
            image_paths.append(path)

    user_prompt = (
        f"User is asking for: '{user_query}'.\n\n"
        f"Here are some related recipes retrieved from our database:\n{recipes}\n"
        f"Using this information and the images from the database suggest some recipes."
    )

    return system_prompt, user_prompt, image_paths

# ============================================================
# 🔹 3️⃣ Generación con Gemini
# ============================================================
def generate_response_gemini(system_prompt, user_prompt, extra_images=None, mime_type="image/png"):
    """
    Versión mejorada de la función de generación con Gemini.
    Incluye manejo de bloqueos, logs de depuración y fallback automático.
    """
    # 🔹 1️⃣ Construir las partes del mensaje multimodal
    parts = [{"text": system_prompt}]

    # 🔹 2️⃣ Añadir imágenes recuperadas desde MinIO (si hay)
    if extra_images and len(extra_images) > 0:
        print(f"🖼️ Adding {len(extra_images)} images from the collection.")
        parts.extend(extra_images)
    else:
        print("⚠️ No images found in collection.")

    # 🔹 3️⃣ Añadir texto del usuario
    parts.append({"text": user_prompt})

    try:
        # 🔹 4️⃣ Llamada directa al modelo (sin 'role')
        response = model.generate_content(
            parts,
            generation_config={"max_output_tokens": 500},
        )

        # 🔹 5️⃣ Mostrar respuesta bruta (debug)
        print("\n=== RAW GEMINI RESPONSE ===")
        print(response)

        # 🔹 6️⃣ Detección de bloqueos de seguridad
        if hasattr(response, "prompt_feedback"):
            fb = response.prompt_feedback
            if getattr(fb, "block_reason", None):
                reason = fb.block_reason
                print(f"⚠️ Gemini blocked the response. Reason: {reason}")
                return f"⚠️ Gemini blocked the response due to safety filters ({reason})."

        # 🔹 7️⃣ Extraer texto generado
        text_output = ""
        if hasattr(response, "candidates") and response.candidates:
            for c in response.candidates:
                if hasattr(c, "content") and hasattr(c.content, "parts"):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            text_output += p.text + "\n"

        if text_output.strip():
            return text_output.strip()

        # 🔹 8️⃣ Si no hay respuesta, reintenta con prompt neutral
        print("⚠️ No response text from Gemini. Retrying with safer prompt...")
        safe_prompt = (
            "You are a friendly cooking assistant. "
            "Please provide safe, helpful recipe suggestions without sensitive terms."
        )
        retry_parts = [{"text": safe_prompt}, {"text": user_prompt}]
        retry_response = model.generate_content(retry_parts)
        text_retry = ""
        if hasattr(retry_response, "candidates") and retry_response.candidates:
            for c in retry_response.candidates:
                if hasattr(c, "content") and hasattr(c.content, "parts"):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            text_retry += p.text + "\n"
        return text_retry.strip() or "⚠️ No response generated even after retry."

    except Exception as e:
        print(f"❌ Error while generating with Gemini: {e}")
        return f"❌ Exception while generating response: {str(e)}"

"""
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Loading Stable Diffusion on {device.upper()}...")

start_time = time.time()

try:
    model_id = "runwayml/stable-diffusion-v1-5" if device == "cpu" else "stabilityai/stable-diffusion-2-1-base"
    print(f"📦 Loading model from cache: {model_id}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True
    )

    print("🔁 Moving model to device...")
    pipe = pipe.to(device)

    print("⚙️  Enabling attention slicing to save memory...")
    pipe.enable_attention_slicing()

    load_time = time.time() - start_time
    print(f"✅ Stable Diffusion loaded successfully on {device.upper()} (took {load_time:.2f} sec)\n")

except Exception as e:
    print(f"❌ Error loading Stable Diffusion: {e}")except Exception as e:
    print(f"❌ Error loading Stable Diffusion: {e}")
    print("⚠️ Tip: If the process exits unexpectedly, you may be running out of memory.")

def generate_image_with_diffusion(prompt, filename="generated_dish.png"):
    
    Genera una imagen usando Stable Diffusion localmente.
    Muestra el prompt y la ruta donde se guarda el archivo.
    
    print("========================================")
    print("🧠 Prompt used to generate the image:")
    print(prompt[:500])
    print("========================================")

     # ⚠️ Stable Diffusion solo admite prompts hasta ~77 tokens (~400 caracteres aprox)
    if len(prompt) > 400:
        print(f"⚠️ Prompt too long ({len(prompt)} chars). Truncating to 400.")
        prompt = prompt[:400]
    # 🔹 Genera la imagen con el modelo
    image = pipe(prompt).images[0]

    # 🔹 Guarda la imagen
    image.save(filename)

    # 🔹 Ruta completa
    abs_path = os.path.abspath(filename)
    print(f"✅ Image saved successfully!")
    print(f"📍 Saved at: {abs_path}")

    return abs_path
"""
# ============================================================
# 🔹 4️⃣ Pipeline completo RAG
# ============================================================

def rag_pipeline(user_query: str, image_path: str):
    # 🔹 1️⃣ Crear embeddings
    text_emb = get_text_embedding(user_query)
    image_emb, _ = get_image_embedding(image_path)

    # 🔹 2️⃣ Recuperar resultados de Chroma (texto + imágenes)
    res_text, res_img = retrieve_from_chroma(col_text, col_img, text_emb, image_emb)

    # 🔹 3️⃣ Construir el prompt textual
    system_prompt, user_prompt, image_paths = build_prompt(user_query, res_text, res_img)

    # 🔹 4️⃣ Descargar imágenes desde MinIO
    extra_images = get_images_from_minio_for_gemini(res_img)

    # 🔹 5️⃣ Generar respuesta con texto + tus imágenes
    response = generate_response_gemini(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_images=extra_images
    )
    #generate_image_from_database(res_text, res_img, user_image_path=image_path)

    return response, image_paths

# ============================================================
# 🔹 6️⃣ Ejecución del script
# ============================================================

if __name__ == "__main__":
    query = "Suggest dishes with pulled pork"
    image_path = r"C:\Users\adals\OneDrive\Documentos\Master\ADSDB-Project\queries\images\pulled_pork.png"

    print("\n Executing RAG pipeline:\n")

    answer, image_paths = rag_pipeline(query, image_path)

    print("\n FINAL RESPONSE FROM GEMINI:")
    print(answer)

    print("\n IMAGES FOUND:")
    for p in image_paths:
        print(f"- {p}")
"""
    print("\n Generating illustrative image from RAG output...")
    generate_image_with_diffusion(answer, filename="dish_from_response.png")
"""