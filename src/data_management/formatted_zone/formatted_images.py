import os
import io
import argparse
from PIL import Image
from datetime import datetime, timezone
from src.common.minio_client import get_minio_client
from minio.error import S3Error

# Buckets
LANDING_BUCKET = "landing-zone"
FORMATTED_BUCKET = "formatted-zone"

# Folders
SRC_PREFIX = "persistent_landing/image_data/"
DST_PREFIX = "formatted/image_data/"

def list_objects(client, bucket, prefix):
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name

def convert_to_png(data: bytes, size=(512, 512)) -> bytes:
    img = Image.open(io.BytesIO(data))
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def dst_key_for(src_key: str) -> str:
    if src_key.startswith(SRC_PREFIX):
        dst_key = src_key.replace(SRC_PREFIX, DST_PREFIX, 1)
    else:
        dst_key = DST_PREFIX + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".png"

def process_image(client, key: str, size=(512, 512)):
    print("Processing:", key)

    # Download
    obj = client.get_object(LANDING_BUCKET, key)
    data = obj.read()
    obj.close(); obj.release_conn()

    # Convert
    png_bytes = convert_to_png(data, size=size)

    # Upload
    dst_key = dst_key_for(key)
    metadata = {
        "x-amz-meta-source-key": key,
        "x-amz-meta-processed-at": datetime.now(timezone.utc).isoformat(),
        "x-amz-meta-format": "png",
    }
    client.put_object(
        FORMATTED_BUCKET,
        dst_key,
        io.BytesIO(png_bytes),
        length=len(png_bytes),
        content_type="image/png",
        metadata=metadata
    )
    print(f"Saved in: {FORMATTED_BUCKET}/{dst_key}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=2, type=int, default=(512, 512))
    args = parser.parse_args()

    client = get_minio_client()

    for key in list_objects(client, LANDING_BUCKET, SRC_PREFIX):
        try:
            process_image(client, key, size=tuple(args.size))
        except S3Error as e:
            print(f"MinIO error with {key}: {e}")
        except Exception as e:
            print(f"Error processing {key}: {e}")

if __name__ == "__main__":
    main()
