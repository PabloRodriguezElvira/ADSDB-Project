from pathlib import Path

# ------------- Bucket names -------------
LANDING_BUCKET = "landing-zone"
FORMATTED_BUCKET = "formatted-zone"
TRUSTED_BUCKET = "trusted-zone"
REJECTED_BUCKET = "rejected-zone"


# ------------- Paths for each zone -------------
FORMATTED_IMAGE_PATH = "formatted/image_data/"
FORMATTED_VIDEO_PATH = "formatted/video_data/"
FORMATTED_TEXT_PATH = "formatted/text_data/"

TRUSTED_IMAGE_PATH = "trusted/image_data/"
TRUSTED_VIDEO_PATH = "trusted/video_data/"
TRUSTED_TEXT_PATH = "trusted/text_data/"

REJECTED_IMAGE_PATH = "rejected/image_data/"
REJECTED_VIDEO_PATH = "rejected/video_data/"
REJECTED_TEXT_PATH = "rejected/text_data/"

LANDING_TEMPORAL_PATH = "temporal_landing/"
LANDING_IMAGE_PATH = "persistent_landing/image_data/"
LANDING_VIDEO_PATH = "persistent_landing/video_data/"
LANDING_TEXT_PATH = "persistent_landing/text_data/"


# ------------- Sub buckets -------------
LANDING_FOLDERS = [
    LANDING_TEMPORAL_PATH,
    LANDING_IMAGE_PATH,
    LANDING_VIDEO_PATH,
    LANDING_TEXT_PATH
]

FORMATTED_FOLDERS = [
    FORMATTED_IMAGE_PATH,
    FORMATTED_VIDEO_PATH,
    FORMATTED_TEXT_PATH
]

TRUSTED_FOLDERS = [
    TRUSTED_IMAGE_PATH,
    TRUSTED_VIDEO_PATH,
    TRUSTED_TEXT_PATH
]

REJECTED_FOLDERS = [
    REJECTED_IMAGE_PATH,
    REJECTED_VIDEO_PATH,
    REJECTED_TEXT_PATH
]


# ------------- Video ingestion -------------
PEXELS_API_KEY = "4tnQhPc8qWhAYxhTjjywT0HIqG4XXELn9mniXCpMJR9xlTcNFx6veL2Y"
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


# ------------- Embeddings and multi-modal tasks -------------
CHROMADB_PATH = Path("embeddings/chromadb").expanduser().resolve() 
TEXT_COLLECTION  = "texts_v1"
IMAGE_COLLECTION = "images_v1"
VIDEO_COLLECTION = "videos_v1"
TEXT_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_QUERY_PATH = "queries/images/"
VIDEO_QUERY_PATH = "queries/videos/"



# ------------- Additional information -------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}
TEXT_EXTENSIONS = {".json", ".txt"}
JSON_NAME = "full_format_recipes.json"


