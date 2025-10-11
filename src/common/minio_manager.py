from io import BytesIO
from minio import Minio
from minio.error import S3Error
from src.common.minio_client import get_minio_client

# Localhost MinIO client
client = get_minio_client()

# Define all the zones and the sub-folders inside them.

LANDING_BUCKET = "landing-zone"
FORMATTED_BUCKET = "formatted-zone"
TRUSTED_BUCKET = "trusted-zone"
REJECTED_BUCKET = "rejected-zone"


LANDING_FOLDERS = [
    "temporal_landing/",
    "persistent_landing/image_data/",
    "persistent_landing/video_data/",
    "persistent_landing/text_data/",
]

FORMATTED_FOLDERS = [
    "formatted/image_data/",
    "formatted/video_data/",
    "formatted/text_data/",
]

TRUSTED_FOLDERS = [
    "trusted/image_data/",
    "trusted/video_data/",
    "trusted/text_data/",
]

REJECTED_FOLDERS = [
    "rejected/image_data/",
    "rejected/video_data/",
    "rejected/text_data/",
]

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
    create_bucket(LANDING_BUCKET)
    for f in LANDING_FOLDERS:
        create_folder(LANDING_BUCKET, f)
        print(f"[OK]: {LANDING_BUCKET}/{f}")

    # Create formatted bucket
    create_bucket(FORMATTED_BUCKET)
    for f in FORMATTED_FOLDERS:
        create_folder(FORMATTED_BUCKET, f)
        print(f"[OK]: {FORMATTED_BUCKET}/{f}")

    # Create trusted bucket
    create_bucket(TRUSTED_BUCKET)
    for f in TRUSTED_FOLDERS:
        create_folder(TRUSTED_BUCKET, f)
        print(f"[OK]: {TRUSTED_BUCKET}/{f}")

    # Create rejected bucket
    create_bucket(REJECTED_BUCKET)
    for f in REJECTED_FOLDERS:
        create_folder(REJECTED_BUCKET, f)
        print(f"[OK]: {REJECTED_BUCKET}/{f}")


if __name__ == "__main__":
    """
    Entry poiny: runs main and handles MinIO errors
    """

    try:
        main()
    except S3Error as e:
        print(f"Error MinIO: {e}")