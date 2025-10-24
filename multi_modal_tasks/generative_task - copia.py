import torch
import numpy as np
from PIL import Image
import os
import io
import google.generativeai as genai
import warnings
from chromadb.utils import embedding_functions as ef
from src.common.chroma_client import get_client, get_text_collection, get_image_collection
from PIL import Image, UnidentifiedImageError
from minio.error import S3Error

from src.common.minio_client import get_minio_client
from src.common.chroma_client import get_client, get_image_collection

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

genai.configure()  # Usa tu GOOGLE_API_KEY del entorno

MODEL_ID = "models/gemini-2.5-flash-lite"  # Puedes cambiar a "models/gemini-2.5-pro" si lo deseas
model = genai.GenerativeModel(MODEL_ID)

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
        f"And also some related images from our collection:\n{images}\n"
        f"Using this information, suggest creative recipes, share cooking techniques, and provide some images."
    )

    return system_prompt, user_prompt, image_paths

# ============================================================
# 🔹 3️⃣ Generación con Gemini
# ============================================================


def generate_image_with_gemini(rag_text, image_paths, output_name="rag_generated_image.png"):
    """
    Genera una imagen ilustrativa combinando el texto del RAG y las imágenes base
    recuperadas desde MinIO.
    """
    print("\n🎨 Generating illustrative image from RAG output...")

    minio_client = get_minio_client()
    bucket_name = "trusted"  # Ajusta si usas otro bucket
    base_images = []

    # 🔹 Descarga imágenes desde MinIO según sus source_key
    for img_key in image_paths:
        try:
            response = minio_client.get_object(bucket_name, img_key)
            img_bytes = response.read()
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            base_images.append(img)
            print(f"✅ Loaded image from MinIO: {img_key}")
        except S3Error as e:
            print(f"⚠️ MinIO error loading {img_key}: {e}")
        except Exception as e:
            print(f"⚠️ Could not process {img_key}: {e}")

    if not base_images:
        print("⚠️ No valid images found in MinIO for generation.")
        return None

    # 🔹 Toma la primera imagen como referencia (puedes combinarlas si quieres)
    ref_image = base_images[0]
    buf = BytesIO()
    ref_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # 🔹 Crea el contenido para Gemini
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": f"Create a realistic and appetizing image inspired by this recipe idea:\n\n{rag_text}"},
                {"inline_data": {"mime_type": "image/png", "data": img_bytes}},
            ],
        }
    ]

    # 🔹 Usa el modelo de generación de imágenes
    image_model = genai.GenerativeModel("models/gemini-2.5-flash-image")
    try:
        response = image_model.generate_content(contents=contents)

        # Si Gemini devuelve datos binarios de imagen
        if response.candidates and response.candidates[0].content.parts:
            part = response.candidates[0].content.parts[0]
            if hasattr(part, "inline_data"):
                out_bytes = part.inline_data.data

                # Guarda la imagen localmente (en la carpeta actual)
                out_path = os.path.join(os.getcwd(), output_name)
                with open(out_path, "wb") as f:
                    f.write(out_bytes)

                print(f"✅ Image generated successfully: {out_path}")
                return out_path

        print("⚠️ Gemini did not return any image.")
    except Exception as e:
        print(f"❌ Error generating image with Gemini: {e}")

    return None
# ============================================================
# 🔹 4️⃣ Pipeline completo RAG
# ============================================================

def rag_pipeline(user_query: str, image_path: str):
    text_emb = get_text_embedding(user_query)
    image_emb, query_image = get_image_embedding(image_path)
    res_text, res_img = retrieve_from_chroma(col_text, col_img, text_emb, image_emb)
    system_prompt, user_prompt, image_paths = build_prompt(user_query, res_text, res_img)
    response = generate_response_gemini(query_image, system_prompt, user_prompt, mime_type="image/png")
    return response, image_paths

# ============================================================
# 🔹 5️⃣ Ejecución del script
# ============================================================

if __name__ == "__main__":
    query = "Give me ideas for dishes with abocado"
    image_path = r"C:\Users\adals\OneDrive\Documentos\Master\ADSDB-Project\pulled_pork.png"

    print("\n Executing RAG pipeline...\n")

    answer, image_paths = rag_pipeline(query, image_path)

    print("\n FINAL RESPONSE FROM GEMINI:")
    print(answer)

    print("\n IMAGES FOUND:")
    for p in image_paths:
        print(f"- {p}")
