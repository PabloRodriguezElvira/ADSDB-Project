from pathlib import Path
from minio.error import S3Error
from src.common.minio_client import get_minio_client
import kagglehub
from datetime import datetime
from zoneinfo import ZoneInfo

# Initialize the MinIO client using a helper function
client = get_minio_client()


def get_json_file_from_folder(folder_path: str, json_file: str):
    """
    Locate a specific JSON file inside a given folder.
    Raises an error if the file does not exist.
    """
    
    file_path = Path(folder_path) / json_file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path
    

def upload_file_to_bucket(bucket: str, destination_object: str, json_file: Path):
    """
    Upload a JSON file to a specified MinIO bucket with metadata.
    Includes ingestion timestamp and data source.
    """

    metadata = {
        "x-amz-meta-data-source": "text",
        "x-amz-meta-ingested-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
    }

    # Upload the file to the given bucket and destination path
    client.fput_object(
        bucket,
        destination_object,
        str(json_file),
        content_type="application/json",
        metadata=metadata,
    )    

    print(f"[OK] Uploaded {json_file} to s3://{bucket}/{destination_object}")


def main():
    """
    Main workflow:
    1. Download the latest version of the 'epirecipes' dataset from Kaggle.
    2. Locate the JSON file inside the downloaded folder.
    3. Upload it to the 'landing-zone' bucket in MinIO.
    """

    # Download dataset from KaggleHub (automatically cached locally)
    path = kagglehub.dataset_download("hugodarwood/epirecipes")

    # Define target bucket and locate the JSON file
    bucket = "landing-zone"
    json_file = get_json_file_from_folder(path, "full_format_recipes.json")

    # Define destination path inside the bucket
    destination_object = f"temporal_landing/{json_file.name}"

    # Upload file to MinIO
    upload_file_to_bucket(bucket, destination_object, json_file)


if __name__ == "__main__":
    """
    Entry point: runs main and handles MinIO errors
    """

    try:
        main()
    except S3Error as e:
        print(f"MinIO Error: {e}")
