import os
import io
import argparse
from typing import Iterable, List, Optional
from PIL import Image, ImageOps
from datetime import datetime
from zoneinfo import ZoneInfo
from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


def list_objects(client, bucket, prefix) -> Iterable[str]:
    """List all objects from a given MinIO bucket and prefix."""
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name


def convert_to_png(data: bytes, size=(512, 512)) -> bytes:
    """Convert an image to a resized PNG with RGBA mode."""
    # Load image from bytes
    img = Image.open(io.BytesIO(data))
    # Correct orientation based on EXIF metadata
    img = ImageOps.exif_transpose(img)

    # Resize and crop image to target size while preserving aspect ratio
    img_resized = ImageOps.fit(
        img,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    )

    # Convert to RGBA and save as PNG in memory
    img_rgba = img_resized.convert("RGBA")
    buf = io.BytesIO()
    img_rgba.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def dst_key_for(src_key: str) -> str:
    """Generate the destination key (path) for the formatted PNG image."""
    if src_key.startswith(config.LANDING_IMAGE_PATH):
        dst_key = src_key.replace(config.LANDING_IMAGE_PATH, config.FORMATTED_IMAGE_PATH, 1)
    else:
        dst_key = config.FORMATTED_IMAGE_PATH + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".png"


def process_image(client, key: str, size=(512, 512), progress: Optional[ProgressBar] = None):
    """Download an image, convert it to PNG (resized), and upload to MinIO."""
    if progress:
        progress.set_description(f"Processing {os.path.basename(key)}", refresh=False)

    # Download image from MinIO 
    obj = client.get_object(config.LANDING_BUCKET, key)
    data = obj.read()
    obj.close()
    obj.release_conn()

    # Convert image to PNG and resize 
    png_bytes = convert_to_png(data, size=size)

    # Upload converted image to formatted bucket 
    dst_key = dst_key_for(key)
    metadata = {
        "x-amz-meta-source-key": key,
        "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        "x-amz-meta-size": f"{size[0]}x{size[1]}",
        "x-amz-meta-format": "png",
    }
    client.put_object(
        config.FORMATTED_BUCKET,
        dst_key,
        io.BytesIO(png_bytes),
        length=len(png_bytes),
        content_type="image/png",
        metadata=metadata
    )


def main():
    """Main entry point: parse arguments, process images, and show progress."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=2, type=int, default=(512, 512))  # Custom image size option
    args = parser.parse_args()

    client = get_minio_client()

    # List all images to process from the landing bucket
    keys: List[str] = list(list_objects(client, config.LANDING_BUCKET, config.LANDING_IMAGE_PATH))

    if not keys:
        print("[WARN] No images found to process.")
        return

    # Process images with a progress bar
    with ProgressBar(
        total=len(keys),
        description="Processing images",
        unit="image",
        unit_scale=False,
    ) as progress:
        for key in keys:
            try:
                process_image(client, key, size=tuple(args.size), progress=progress)
            except S3Error as e:
                progress.write(f"MinIO error with {key}: {e}")
            except Exception as e:
                progress.write(f"Error processing {key}: {e}")
            finally:
                progress.update(1)


if __name__ == "__main__":
    main()
