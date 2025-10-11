import os, chromadb
from chromadb.utils import embedding_functions as ef

DB_PATH = os.getenv("DB_PATH", "exploitation_zone/chromadb")

TEXT_COLLECTION  = os.getenv("TEXT_COLLECTION",  "texts_v1")
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION", "images_v1")

TEXT_MODEL_NAME  = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

def get_client():
    os.makedirs(DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=DB_PATH)

def get_text_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=TEXT_COLLECTION,
        embedding_function=ef.SentenceTransformerEmbeddingFunction(model_name=TEXT_MODEL_NAME),
        metadata={"format":"text"}
    )

def get_image_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        embedding_function=ef.OpenCLIPEmbeddingFunction(),
        metadata={"format":"image"}
    )
