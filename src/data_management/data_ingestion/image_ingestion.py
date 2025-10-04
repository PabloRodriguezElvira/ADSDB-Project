from pathlib import Path
from zipfile import ZipFile
from minio import Minio
from minio.error import S3Error
from src.common.minio_client import get_minio_client 
from datetime import datetime
from zoneinfo import ZoneInfo
import kagglehub
import mimetypes


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}



def upload_file_to_bucket(client: Minio, bucket: str, object_name: str, local_file: Path, img_type: str):
    metadata = {
        "x-amz-meta-data-source": "image",
        "x-amz-meta-image-type": img_type,
        "x-amz-meta-ingested-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
    }

    # Get the type of the image
    content_type, _ = mimetypes.guess_type(local_file.name)

    client.fput_object(
        bucket,
        object_name,
        str(local_file),
        content_type=content_type,
        metadata=metadata,
    )
    print(f"[OK] Subido {local_file} a s3://{bucket}/{object_name}")



def upload_directory_images(client: Minio, bucket: str, folder: str, dataset_path: Path):
    some_image_uploaded = False
    img_counter = 0

    for split_dir in dataset_path.iterdir():
        # Traverse all subdirectories and files of split_dir. Split_dir is one of these folders: evaluation, training or validation.
        for img_file in split_dir.rglob("*"):
            # If it's a folder, we skip it.
            if not img_file.is_file():
                continue

            # Image case
            if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                img_type = img_file.parent.name
                img_name = f"{img_type}-{split_dir.name}-{img_counter}{img_file.suffix}"
                object_name = f"{folder}/{img_name}"
                upload_file_to_bucket(client, bucket, object_name, img_file, img_type)
                img_counter += 1
                some_image_uploaded = True

    return some_image_uploaded 


def main():
    client = get_minio_client()

    path = kagglehub.dataset_download("trolukovich/food11-image-dataset")
    dataset_path = Path(path)

    bucket = "landing-zone"
    folder = "temporal_landing"

    # Upload images to temporal_landing bucket
    uploaded = upload_directory_images(client, bucket, folder, dataset_path)

    if not uploaded:
        print("[WARN] No se encontraron imágenes para subir al bucket temporal.")


if __name__ == "__main__":
    try:
        main()
    except S3Error as e:
        print(f"Error MinIO: {e}")

