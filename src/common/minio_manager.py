from io import BytesIO
from minio import Minio
from minio.error import S3Error
from src.common.minio_client import get_minio_client

# Localhost MinIO client
client = get_minio_client()

LANDING_BUCKET = "landing-zone"

FOLDERS = [
    "temporal_landing/",
    "persistent_landing/",
    "persistent_landing/image_data/",
    "persistent_landing/video_data/",
    "persistent_landing/text_data/",
]


def create_landing_bucket():
    if not client.bucket_exists(LANDING_BUCKET):
        client.make_bucket(LANDING_BUCKET)


def create_folder(folder: str):
    # Folders must end with "/"
    if not folder.endswith("/"):
        folder += "/"

    # The object we put inside the bucket is an empty folder
    client.put_object(
        LANDING_BUCKET,
        folder,
        data=BytesIO(b""),
        length=0,
        content_type="application/octet-stream",
    )


def main():
    create_landing_bucket()
    for f in FOLDERS:
        create_folder(f)
        print(f"[OK]: {LANDING_BUCKET}/{f}")



if __name__ == "__main__":
    try:
        main()
    except S3Error as e:
        print(f"Error MinIO: {e}")
