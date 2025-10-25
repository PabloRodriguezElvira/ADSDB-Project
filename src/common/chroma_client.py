import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions as ef

import src.common.global_variables as config

# Database path and collection names
DB_PATH = Path(os.getenv("DB_PATH", "embeddings/chromadb")).expanduser().resolve()
TEXT_COLLECTION  = os.getenv("TEXT_COLLECTION", config.TEXT_COLLECTION)
IMAGE_COLLECTION = os.getenv("IMAGE_COLLECTION", config.IMAGE_COLLECTION)
VIDEO_COLLECTION = os.getenv("VIDEO_COLLECTION", config.VIDEO_COLLECTION)
TEXT_MODEL_NAME  = os.getenv("TEXT_MODEL_NAME", config.TEXT_MODEL_NAME)

# Create the embeddings directory if it doesn't exist
DB_PATH.mkdir(parents=True, exist_ok=True)

# Embedding functions for text and images/videos
_text_ef = ef.SentenceTransformerEmbeddingFunction(model_name=TEXT_MODEL_NAME)
_image_ef = ef.OpenCLIPEmbeddingFunction()


def get_client():
    """
    Returns a Chroma client.
    """
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=str(DB_PATH))
    else:
        return chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(DB_PATH)
        ))


def get_text_collection(client=None):
    """
    Returns the text embedding collection.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=TEXT_COLLECTION,
        embedding_function=_text_ef,
        metadata={"format": "text"}
    )


def get_image_collection(client=None):
    """
    Returns the image embedding collection.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=IMAGE_COLLECTION,
        embedding_function=_image_ef,
        metadata={"format": "image"}
    )


def get_video_collection(client=None):
    """
    Returns the video embedding collection.
    """
    client = client or get_client()
    return client.get_or_create_collection(
        name=VIDEO_COLLECTION,
        embedding_function=_image_ef,
        metadata={"format": "video"}
    )
