from pathlib import Path
from zipfile import ZipFile
from minio import Minio
from minio.error import S3Error
from src.common.minio_client import get_minio_client 
import kagglehub


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}



def upload_file_to_bucket(client: Minio, bucket: str, object_name: str, local_file: Path):
    client.fput_object(bucket, object_name, str(local_file))
    print(f"[OK] Subido {local_file} a s3://{bucket}/{object_name}")



def upload_directory_images(client: Minio, bucket: str, folder: str, dataset_path: Path):
    uploaded = False

    for split_dir in dataset_path.iterdir():
        # Traverse all subdirectories and files of split_dir. Split_dir is one of these folders: evaluation, training or validation.
        for img_file in split_dir.rglob("*"):
            if not img_file.is_file():
                continue

            if img_file.suffix.lower() in IMAGE_EXTENSIONS:
                print(img_file)
                object_name = f"{folder}/{img_file.name}"
                print(object_name)
                upload_file_to_bucket(client, bucket, object_name, img_file)
                uploaded = True

    return uploaded 


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



