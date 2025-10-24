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

MODEL_ID =  "models/gemini-2.5-flash"     # Puedes cambiar a "models/gemini-2.5-pro" si lo deseas
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
        f"Using this information suggest cooking techniques provide some images."
    )

    return system_prompt, user_prompt, image_paths

# ============================================================
# 🔹 3️⃣ Generación con Gemini
# ============================================================

def generate_response_gemini(query_image, system_prompt, user_prompt, mime_type="image/png"):
    buf = io.BytesIO()
    fmt = "PNG" if mime_type.endswith("png") else "JPEG"
    query_image.save(buf, format=fmt)
    image_bytes = buf.getvalue()
    

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": system_prompt},
                {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
                {"text": user_prompt},
            ],
        }
    ]

    try:
        response = model.generate_content(
            contents=contents,
            generation_config={"max_output_tokens": 200},
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
    except Exception as e:
        print(f" Error calling Gemini: {e}")
        return "Gemini request failed."

    # 🔹 Extrae texto manualmente (sin .text)
    text_parts = []
    if hasattr(response, "candidates") and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "text"):
                        text_parts.append(part.text)

    if text_parts:
        return "\n".join(text_parts)

    # 🚨 Si no hay texto, imprime diagnóstico y reformula
    print("Gemini did not return valid text.")
    if hasattr(response, "candidates") and response.candidates:
        c = response.candidates[0]
        print("🔍 Finish reason:", getattr(c, "finish_reason", "unknown"))
        print("🔍 Safety ratings:", getattr(c, "safety_ratings", "none"))
    print("🪶 Prompt feedback:", getattr(response, "prompt_feedback", "none"))

    blocked_terms = ["pork", "bacon", "ham"]
    safe_prompt = user_prompt
    for term in blocked_terms:
        safe_prompt = safe_prompt.replace(term, "slow-cooked shredded meat")

    if safe_prompt != user_prompt:
        print("♻️ Reformulating prompt to avoid safety block...")
        return generate_response_gemini(query_image, system_prompt, safe_prompt, mime_type)

    return "Gemini blocked this request."


"""
def generate_image_with_gemini(base_images, text_prompt, output_name="rag_generated_image.png"):

    Genera una nueva imagen combinando imágenes de referencia + texto del RAG.
    Soporta imágenes locales o almacenadas en MinIO.

    Args:
        base_images (list[str]): rutas locales o rutas en MinIO (ej. "trusted/image_data/...").
        text_prompt (str): descripción generada por el RAG.
        output_name (str): nombre del archivo de salida.

    Returns:
        str | None: ruta local del archivo generado o None si falla.

    genai.configure()
    image_model = genai.GenerativeModel("models/gemini-2.5-flash-image")
    
    from src.common.minio_client import get_minio_client
    client_s3 = get_minio_client()
    image_parts = []

    for path in base_images:
        img = None
        try:
            # --- 🔹 Detectar si la imagen viene de MinIO o del sistema local ---
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
            else:
                # Si no existe localmente, intenta descargar desde MinIO
                bucket = config.TRUSTED_BUCKET
                # Normaliza el nombre del objeto (quita prefijos incorrectos)
                object_name = path.replace("trusted/", "").lstrip("/")
                response = client_s3.get_object(bucket, object_name)
                img = Image.open(io.BytesIO(response.read())).convert("RGB")
                response.close()
                response.release_conn()

            # --- 🔹 Convertir a bytes PNG ---
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            image_parts.append({"inline_data": {"mime_type": "image/png", "data": image_bytes}})

        except Exception as e:
            print(f"⚠️ No se pudo procesar {path}: {e}")

    if not image_parts:
        print("⚠️ No valid images found for generation.")
        return None

    # --- 🔹 Construir prompt multimodal ---
    contents = [{
        "role": "user",
        "parts": [
            *image_parts,
            {"text": f"Using these reference images, create a realistic and appetizing photo illustrating: {text_prompt}"}
        ]
    }]

    # --- 🔹 Definir ruta de salida ---
    base_dir = os.path.dirname(base_images[0]) if base_images else os.getcwd()
    out_path = os.path.join(base_dir, output_name)

    # --- 🔹 Llamar a Gemini ---
    try:
        response = image_model.generate_content(contents=contents)

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                    out_bytes = part.inline_data.data
                    with open(out_path, "wb") as f:
                        f.write(out_bytes)
                    print(f"✅ Image generated successfully: {out_path}")
                    return out_path

        print("⚠️ Gemini did not return any image.")
        return None

    except Exception as e:
        print(f"❌ Error generating image with Gemini: {e}")
        return None
"""
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
# 🔹 6️⃣ Ejecución del script
# ============================================================

if __name__ == "__main__":
    query = "Suggest dishes with slow-cooked shredded meat"
    image_path = r"C:\Users\adals\OneDrive\Documentos\Master\ADSDB-Project\avocado.jpg"

    print("\n Executing RAG pipeline...\n")

    answer, image_paths = rag_pipeline(query, image_path)

    print("\n FINAL RESPONSE FROM GEMINI:")
    print(answer)

    print("\n IMAGES FOUND:")
    for p in image_paths:
        print(f"- {p}")

    print("\n Generating illustrative image from RAG output...")
    #generate_image_with_gemini(image_paths, answer)