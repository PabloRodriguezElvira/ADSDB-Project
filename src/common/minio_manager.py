from io import BytesIO
from minio import Minio
from minio.error import S3Error
from src.common.minio_client import get_minio_client
import src.common.global_variables as config

# Localhost MinIO client
client = get_minio_client()

def create_bucket(bucket: str):
    """
    Create the bucket if it does not exist.
    """

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def create_folder(bucket: str, folder: str):
    """
    Creates the folder inside the bucker
    """

    # Folders must end with "/"
    if not folder.endswith("/"):
        folder += "/"

    # The object we put inside the bucket is an empty folder
    client.put_object(
        bucket,
        folder,
        data=BytesIO(b""),
        length=0,
        content_type="application/octet-stream",
    )



def main():
    """
    Creates the buckets with all their sub buckets
    """

    # Create landing bucket
    create_bucket(config.LANDING_BUCKET)
    for f in config.LANDING_FOLDERS:
        create_folder(config.LANDING_BUCKET, f)
        print(f"[OK]: {config.LANDING_BUCKET}/{f}")

    # Create formatted bucket
    create_bucket(config.FORMATTED_BUCKET)
    for f in config.FORMATTED_FOLDERS:
        create_folder(config.FORMATTED_BUCKET, f)
        print(f"[OK]: {config.FORMATTED_BUCKET}/{f}")

    # Create trusted bucket
    create_bucket(config.TRUSTED_BUCKET)
    for f in config.TRUSTED_FOLDERS:
        create_folder(config.TRUSTED_BUCKET, f)
        print(f"[OK]: {config.TRUSTED_BUCKET}/{f}")

    # Create rejected bucket
    create_bucket(config.REJECTED_BUCKET)
    for f in config.REJECTED_FOLDERS:
        create_folder(config.REJECTED_BUCKET, f)
        print(f"[OK]: {config.REJECTED_BUCKET}/{f}")


if __name__ == "__main__":
    """
    Entry poiny: runs main and handles MinIO errors
    """

    try:
        main()
    except S3Error as e:
        print(f"Error MinIO: {e}")