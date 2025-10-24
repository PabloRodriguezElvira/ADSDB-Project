import torch
import numpy as np
from PIL import Image
import minio
import os
import io
import google.generativeai as genai
import warnings
from chromadb.utils import embedding_functions as ef
from src.common.chroma_client import get_client, get_text_collection, get_image_collection
from minio.error import S3Error

from src.common.minio_client import get_minio_client
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

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
text_ef = ef.SentenceTransformerEmbeddingFunction(model_name=TEXT_MODEL_NAME)
image_ef = ef.OpenCLIPEmbeddingFunction()

genai.configure(api_key="AIzaSyD8De0Y6Dqy19AHe-Kmd549uNRaqtbll6g")  # Usa tu GOOGLE_API_KEY del entorno

MODEL_ID =  "models/gemini-2.5-flash"     # Puedes cambiar a "models/gemini-2.5-pro" si lo deseas
model = genai.GenerativeModel(MODEL_ID)
image_model = genai.GenerativeModel("models/gemini-2.5-flash-image")
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


import src.common.global_variables as config  # 👈 para acceder a TRUSTED_BUCKET

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
        f"Using this information and  images suggest some recipes."
    )

    return system_prompt, user_prompt, image_paths

# ============================================================
# 🔹 3️⃣ Generación con Gemini
# ============================================================
def generate_response_gemini(system_prompt, user_prompt, extra_images=None, mime_type="image/png"):
    """
    Genera una respuesta con Gemini usando:
      - texto (system_prompt + user_prompt)
      - imágenes recuperadas desde tu base de datos (extra_images)
    No usa la imagen del usuario.
    """

    # 🔹 1️⃣ Construir las partes del mensaje multimodal
    parts = [{"text": system_prompt}]

    # 🔹 2️⃣ Añadir las imágenes recuperadas desde MinIO
    if extra_images and len(extra_images) > 0:
        print(f" Adding the {len(extra_images)} images selected from the collection.")
        parts.extend(extra_images)
    else:
        print("We could not find images.")

    # 🔹 3️⃣ Añadir el texto del usuario al final
    parts.append({"text": user_prompt})

    # 🔹 4️⃣ Contenido final
    contents = [{"role": "user", "parts": parts}]

    # 🔹 5️⃣ Llamada al modelo Gemini
    response = model.generate_content(
        contents=contents,
        generation_config={"max_output_tokens": 500},
    )

    # 🔹 6️⃣ Extraer texto generado
    text_output = ""
    if hasattr(response, "candidates") and response.candidates:
        for c in response.candidates:
            if hasattr(c, "content") and hasattr(c.content, "parts"):
                for p in c.content.parts:
                    if hasattr(p, "text"):
                        text_output += p.text + "\n"

    return text_output.strip() or " No response generated by Gemini."



def generate_image_from_database(res_text, res_img, user_image_path=None, output_name="db_generated_image.png"):
    """
    Genera una imagen usando:
      - los textos recuperados desde tu base de datos (res_text)
      - las imágenes recuperadas desde MinIO (res_img)
      - y opcionalmente la imagen del usuario (user_image_path)
    Usa el modelo Gemini-2.5-flash-image.
    """


    # 🔹 1️⃣ Extraer texto desde tu base (res_text)
    db_text = ""
    for doc, meta in zip(res_text.get("documents", [[]])[0], res_text.get("metadatas", [[]])[0]):
        title = meta.get("title", "") if meta else ""
        snippet = doc.strip().replace("\n", " ")
        db_text += f"{title}\n{snippet}\n\n"

    if not db_text.strip():
        db_text = "Cooking recipes retrieved from database."

    # 🔹 2️⃣ Obtener imágenes recuperadas desde MinIO
    base_images = get_images_from_minio_for_gemini(res_img)
    print(f"📦 {len(base_images)} imágenes recuperadas desde MinIO para Gemini.")

    # 🔹 3️⃣ Añadir la imagen del usuario (opcional)
    if user_image_path and os.path.exists(user_image_path):
        try:
            img = Image.open(user_image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            base_images.append({"inline_data": {"mime_type": "image/png", "data": image_bytes}})
            print("👤 Imagen del usuario añadida como referencia visual.")
        except Exception as e:
            print(f"⚠️ No se pudo procesar la imagen del usuario: {e}")

    if not base_images:
        print("⚠️ No hay imágenes disponibles para generar una nueva.")
        return None

    # 🔹 4️⃣ Construir prompt textual basado solo en los textos de tu base
    prompt_text = (
        "Using the following recipes and the reference images provided, "
        "generate a realistic, appetizing, high-quality image that visually represents these dishes:\n\n"
        f"{db_text}"
    )

    contents = [{
        "role": "user",
        "parts": [
            *base_images,
            {"text": prompt_text},
        ]
    }]

    # 🔹 5️⃣ Llamar al modelo Gemini
    try:
        response = image_model.generate_content(contents=contents)

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                    out_bytes = part.inline_data.data
                    out_path = os.path.join(os.getcwd(), output_name)
                    with open(out_path, "wb") as f:
                        f.write(out_bytes)
                    print(f"✅ Imagen generada exitosamente: {out_path}")
                    return out_path

        print("⚠️ Gemini no devolvió una imagen.")
        return None

    except Exception as e:
        print(f"❌ Error generando imagen con Gemini: {e}")
        return None


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
    generate_image_from_database(res_text, res_img, user_image_path=image_path)

    return response, image_paths

# ============================================================
# 🔹 6️⃣ Ejecución del script
# ============================================================

if __name__ == "__main__":
    query = "Suggest dishes with pulled pork"
    image_path = r"C:\Users\adals\OneDrive\Documentos\Master\ADSDB-Project\pulled_pork.png"

    print("\n Executing RAG pipeline:\n")

    answer, image_paths = rag_pipeline(query, image_path)

    print("\n FINAL RESPONSE FROM GEMINI:")
    print(answer)

    print("\n IMAGES FOUND:")
    for p in image_paths:
        print(f"- {p}")

    #print("\n Generating illustrative image from RAG output...")
    #generate_image_with_gemini(image_paths, answer)