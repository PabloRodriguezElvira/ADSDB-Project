from __future__ import annotations
from pathlib import Path
import google.generativeai as genai
import src.common.global_variables as config  
import numpy as np
from PIL import Image
import os
import google.generativeai as genai
import warnings
from src.common.minio_client import get_minio_client
#from src.multi_modal_tasks.same_modality_task import _encode_text_input,
from src.common.chroma_client import (
    get_client,
    get_image_collection,
    get_text_collection,
    _text_image_ef,
    GENERATIVE_MODEL
)

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_CPP_PLUGIN_LOGGER"] = "NONE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Minio connection
client_s3 = get_minio_client()

# Cache Chroma connections
client = get_client()
col_text = get_text_collection(client)
col_img = get_image_collection(client)

BASE_DIR = Path(__file__).resolve().parents[2]  # Project root

# Selection of the generative model and api configuration
genai.configure(api_key=config.GOOGLE_API_KEY)
model = genai.GenerativeModel(GENERATIVE_MODEL)


# silenced warnings
warnings.filterwarnings(
    "ignore",
    message=r"resume_download is deprecated",
    category=FutureWarning,
    module="huggingface_hub",
)

def _encode_text_input(text: str) -> List[float]:
    """Return the embedding vector for the provided text using the ChromaDB function."""
    clean = text.strip()
    if not clean:
        raise ValueError("Text query must be a non-empty string.")

    embeddings = _text_image_ef([clean])
    if not embeddings:
        raise ValueError("Unable to compute embedding for the provided text.")
    return embeddings[0]

def get_image_embedding(image_path: str):
    """Function to get the embedding from the input image, not equivalent to the previous tasks"""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    emb_list = _text_image_ef([arr])
    return emb_list[0]

def retrieve_from_chroma(col_text, col_img, text_emb, image_emb, k_text=2, k_img=2):
    """Function to make the search (text to text and image to image)."""
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
    """Retrieve the most similar images from the Chroma collection and fetch their actual binary data 
    from the TRUSTED_BUCKET in MinIO. Since Chroma only stores embeddings and metadata (not raw files), 
    the raw image bytes must be downloaded directly from MinIO to be used by the generative model.
    """
    image_parts = []

    for meta in res_img.get("metadatas", [[]])[0]:
        source_key = meta.get("source_key")
        if not source_key:
            continue

        try:
            # Bucket from which we will obtain the information we need.
            bucket = config.TRUSTED_BUCKET  
            mime_type = "image/png"        

            response = client_s3.get_object(bucket, source_key)
            data = response.read() # Read the bytes. 
            response.close()
            response.release_conn()


            # Store the information.
            image_parts.append({"inline_data": {"mime_type": mime_type, "data": data}})

        except Exception as e:
            print(f" Error descargando {source_key} desde MinIO ({bucket}): {e}")

    return image_parts

def build_prompt(user_query, res_text, res_img):
    "Build the system prompt and user_prompt."
    system_prompt = (
        "You are a chef sharing cooking idea using the text and images found in a friendly tone." )
    
    recipes = "" # From the text obtained we store it as a summarized context for the generative mode.
    for doc, meta in zip(res_text["documents"][0], res_text["metadatas"][0]):
        title = doc[:20]
        content = doc[20:]
        recipes += f"- {title}:\n  {content[:250]}...\n\n"

    image_paths = [] # In order to return which images from our collection we found
    for meta in res_img["metadatas"][0]:
        path = meta.get("source_key", "")
        if path:
            image_paths.append(path)

    user_prompt = ( # Build the user prompt
        f"User is asking for: '{user_query}'.\n\n"
        f"Here are some related recipes retrieved from our database:\n{recipes}\n"
        f"Using this information and the images from the database suggest some recipes."
    )

    return system_prompt, user_prompt, image_paths


def generate_response_gemini(system_prompt, user_prompt, images_found=None, mime_type="image/png"):
    """
    Generate the multimodal response using the Gemini model. Using the system prompt
    and user prompt adding the similar images found in the collection.
    It containd handle safety blocks in order to generate a response with a neutral prompt 
    if needed, and returns the generated text output.
    """
    # Build the multimodal message
    parts = [{"text": system_prompt}]

    # Adding the images found from minIO
    if images_found and len(images_found) > 0:
        print(f" Adding {len(images_found)} images from the collection.")
        parts.extend(images_found)
    else:
        print("No images found in collection.")

    # Add the text from the user.
    parts.append({"text": user_prompt})

    try:
        # Call the model.
        response = model.generate_content(
            parts,
            generation_config={"max_output_tokens": 500},
        )

        # Show the response.
        print("\n GEMINI RESPONSE:")
        print(response)

        # Detect security blocks.
        if hasattr(response, "prompt_feedback"):
            fb = response.prompt_feedback
            if getattr(fb, "block_reason", None):
                reason = fb.block_reason
                print(f"Gemini blocked the response. Reason: {reason}")
                return f"Gemini blocked the response due to safety filters ({reason})."

        # Extract the generated block text from GEMINI.
        text_output = ""
        if hasattr(response, "candidates") and response.candidates:
            for c in response.candidates:
                if hasattr(c, "content") and hasattr(c.content, "parts"):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            text_output += p.text + "\n"

        # If GEMINI generated text returned it, if not try with a safer prompt.
        if text_output.strip():
            return text_output.strip()

        # If GEMINI didn't return anything, try with a safer prompt
        print(" No response text from Gemini. Retrying with safer prompt")
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
        return text_retry.strip() or " No response generated even after retry."

    except Exception as e:
        print(f" Error while generating with Gemini: {e}")
        return f" Exception while generating response: {str(e)}"


def rag_pipeline(user_query: str, image_path: str):
    """Combine the above functions to generate an answer."""
    # Generate the embeddings.
    text_emb = _encode_text_input(user_query)
    image_emb = get_image_embedding(image_path)

    # Get the text and images most similar from chroma.
    res_text, res_img = retrieve_from_chroma(col_text, col_img, text_emb, image_emb)

    # Build the textual prompt
    system_prompt, user_prompt, image_paths = build_prompt(user_query, res_text, res_img)

    # Download images from minIO
    images_found = get_images_from_minio_for_gemini(res_img)

    # Generate the answer with text and images
    response = generate_response_gemini(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images_found=images_found
    )
    #generate_image_from_database(res_text, res_img, user_image_path=image_path)

    return response, image_paths

if __name__ == "__main__":
    """Main entry point: entering as input the query user and the image"""
    query = "Suggest dishes with avocado"
    image = "avocado.jpg"
    image_path = (BASE_DIR / f"{config.IMAGE_QUERY_PATH}{image}").resolve()
    print("\n Executing RAG pipeline:\n")

    answer, image_paths = rag_pipeline(query, image_path)

    print("\n FINAL RESPONSE FROM GEMINI:")
    print(answer)

    print("\n IMAGES FOUND:")
    for p in image_paths:
        print(f"- {p}")
