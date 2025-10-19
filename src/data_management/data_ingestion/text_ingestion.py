from pathlib import Path
import kagglehub
from datetime import datetime
from zoneinfo import ZoneInfo
from minio.error import S3Error

from src.common.minio_client import get_minio_client
from src.common.progress_bar import ProgressBar
import src.common.global_variables as config

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

    file_size = json_file.stat().st_size

    # Use an instance of our ProgressBar class to pass it to the MinIO client function fput_object.
    with ProgressBar(
        total=file_size,
        description=f"Uploading {json_file.name}",
    ) as progress:
        client.fput_object(
            bucket,
            destination_object,
            str(json_file),
            content_type="application/json",
            metadata=metadata,
            progress=progress,
        )

def main():
    """
    Main workflow:
    1. Download the latest version of the 'epirecipes' dataset from Kaggle.
    2. Locate the JSON file inside the downloaded folder.
    3. Upload it to the 'landing-zone' bucket in MinIO.
    """

    # Download dataset from KaggleHub (automatically cached locally)
    dataset_path = kagglehub.dataset_download("hugodarwood/epirecipes")

    # Define target bucket and locate the JSON file
    json_file = get_json_file_from_folder(dataset_path, config.JSON_NAME)

    # Define destination path inside the bucket
    destination_object = f"{config.LANDING_TEMPORAL_PATH}{json_file.name}"

    # Upload file to MinIO
    upload_file_to_bucket(config.LANDING_BUCKET, destination_object, json_file)


if __name__ == "__main__":
    """
    Entry point: runs main and handles MinIO errors
    """

    try:
        main()
    except S3Error as e:
        print(f"MinIO Error: {e}")
