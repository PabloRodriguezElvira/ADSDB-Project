from pathlib import Path
from zipfile import ZipFile
from minio import Minio
from minio.error import S3Error
from datetime import datetime
from zoneinfo import ZoneInfo
import kagglehub
import mimetypes
from typing import Optional

from src.common.minio_client import get_minio_client 
from src.common.progress_bar import ProgressBar
import src.common.global_variables as config


def upload_file_to_bucket(
    client: Minio,
    bucket: str,
    object_name: str,
    local_file: Path,
    img_type: str,
    progress: Optional[ProgressBar] = None,
):
    """
    Upload a single image file to the specified MinIO bucket.
    Adds data source, image type, and ingestion timestamp as metadata.
    """

    metadata = {
        "x-amz-meta-data-source": "image",
        "x-amz-meta-image-type": img_type,
        "x-amz-meta-ingested-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
    }

    # Determine MIME type. This is a standard that indicates the type of content of a file.
    content_type, _ = mimetypes.guess_type(local_file.name)

    # Create the progress bar only if it has not been created before. It shows the uploading progress of all images. 
    if progress is None:
        file_size = local_file.stat().st_size

        with ProgressBar(
            total=file_size,
            description=f"Uploading {local_file.name}",
        ) as single_progress:
            client.fput_object(
                bucket,
                object_name,
                str(local_file),
                content_type=content_type,
                metadata=metadata,
                progress=single_progress,
            )
    else:
        client.fput_object(
            bucket,
            object_name,
            str(local_file),
            content_type=content_type,
            metadata=metadata,
            progress=progress,
        )


def upload_directory_images(client: Minio, bucket: str, folder: str, dataset_path: Path):
    """
    Traverse the dataset directory and upload all image files to the given bucket/folder.
    Generates unique object names based on image type, split (train/val/test), and image counter.
    """

    upload_queue = []
    total_size = 0

    # First pass: gather images and compute total size for aggregate progress
    for split_dir in dataset_path.iterdir():
        # Traverse subdirectories (evaluation, training, validation)
        for img_file in split_dir.rglob("*"):
            if not img_file.is_file():
                continue

            if img_file.suffix.lower() in config.IMAGE_EXTENSIONS:
                img_type = img_file.parent.name
                file_size = img_file.stat().st_size
                upload_queue.append((split_dir.name, img_file, img_type, file_size))
                total_size += file_size

    if not upload_queue:
        return False

    with ProgressBar(
        total=total_size,
        description="Uploading image dataset",
    ) as progress:
        for idx, (split_name, img_file, img_type, _) in enumerate(upload_queue):
            img_name = f"{img_type}-{split_name}-{idx}{img_file.suffix}"
            object_name = f"{folder}{img_name}"
            upload_file_to_bucket(client, bucket, object_name, img_file, img_type, progress)

    return True


def main():
    """
    Download the Food11 dataset from Kaggle, then upload its images
    to the MinIO temporal landing bucket (landing-zone/temporal_landing).
    """

    client = get_minio_client()

    path = kagglehub.dataset_download("trolukovich/food11-image-dataset")
    print(path)
    dataset_path = Path(path)
    

    # Upload images to temporal_landing bucket
    uploaded = upload_directory_images(client, config.LANDING_BUCKET, config.LANDING_TEMPORAL_PATH, dataset_path)

    if not uploaded:
        print("[WARN] No se encontraron imágenes para subir al bucket temporal.")


if __name__ == "__main__":
    """
    Entry poiny: runs main and handles MinIO errors
    """

    try:
        main()
    except S3Error as e:
        print(f"Error MinIO: {e}")
