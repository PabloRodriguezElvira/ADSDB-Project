import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions as ef

DB_PATH = Path(os.getenv("DB_PATH", "embeddings/chromadb")).expanduser().resolve() #Escoger el path
TEXT_COLLECTION  = os.getenv("TEXT_COLLECTION",  "texts_v1")
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION", "images_v1")
TEXT_MODEL_NAME  = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

DB_PATH.mkdir(parents=True, exist_ok=True)

_text_ef = ef.SentenceTransformerEmbeddingFunction(model_name=TEXT_MODEL_NAME)
_image_ef = ef.OpenCLIPEmbeddingFunction()

def get_client():
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=str(DB_PATH))
    else:
        return chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(DB_PATH)
        ))

def get_text_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=TEXT_COLLECTION,
        embedding_function=_text_ef,
        metadata={"format":"text"}
    )

def get_image_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        embedding_function=_image_ef,
        metadata={"format":"image"}
    )
