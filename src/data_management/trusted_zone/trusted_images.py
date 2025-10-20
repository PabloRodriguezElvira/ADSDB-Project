import os
import io
import re
import hashlib
from datetime import datetime
from typing import Iterable, List
from zoneinfo import ZoneInfo
from minio.error import S3Error
from PIL import Image, ImageStat, UnidentifiedImageError

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


def list_objects(client, bucket, prefix) -> Iterable[str]:
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name


def dst_key_for(src_key: str, dst_prefix: str):
    if src_key.startswith(config.FORMATTED_IMAGE_PATH):
        dst_key = src_key.replace(config.FORMATTED_IMAGE_PATH, dst_prefix, 1)
    else:
        dst_key = os.path.join(dst_prefix, os.path.basename(src_key))
    return dst_key


# checking out if we are able to open all the images
def is_image_valid(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


# check image properties
def image_properties(data: bytes, expected_size=(512, 512)):
    result = {
        "format": None,
        "size": None,
        "color": None,
        "size_ok": False,
        "color_ok": False,
        "format_ok": False,
        "error": None,
    }
    # let's check if the images meet the expected characteristics
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()

            result["format"] = img.format
            result["size"] = img.size
            result["color"] = img.mode

            if img.size == expected_size:
                result["size_ok"] = True

            if img.mode in ("RGB", "RGBA"):
                result["color_ok"] = True

            if img.format == "PNG":
                result["format_ok"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# Image properties
def brightness_and_contrast(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("L")  # grey scale
    stat = ImageStat.Stat(img)  # create an object with image features
    brightness = stat.mean[0] / 255.0  # normalized mean brightness
    contrast = stat.stddev[0] / 255.0  # normalized contrast

    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "brightness_ok": 0.2 <= brightness <= 0.8,
        "contrast_ok": contrast >= 0.2,
    }


# Combine validation functions
def validate_images(images: dict):
    valid_images = {}
    invalid_images = {}

    for name, data in images.items():
        if not is_image_valid(data):
            invalid_images[name] = "Corrupt or unreadable"
            continue

        props = image_properties(data)
        bright = brightness_and_contrast(data)

        if (
            props["size_ok"]
            and props["format_ok"]
            and props["color_ok"]
            and bright["brightness_ok"]
            and bright["contrast_ok"]
        ):
            valid_images[name] = data
        else:
            reasons = []
            if not props["size_ok"]:
                reasons.append("Invalid size")
            if not props["format_ok"]:
                reasons.append("Not PNG")
            if not props["color_ok"]:
                reasons.append("Not RGB/RGBA")
            if not bright["brightness_ok"]:
                reasons.append("Bad brightness")
            if not bright["contrast_ok"]:
                reasons.append("Low contrast")
            invalid_images[name] = ", ".join(reasons)

    return {"valid": valid_images, "invalid": invalid_images}


def duplicates(images: dict):
    
    md5_map = {}# md5:name of unique images
    duplicates = []# the names of the images in which its hash is already in md5_map
    hashes = {}# names of name:hash 

    for name, data in images.items():
        md5 = hashlib.md5(data).hexdigest()
        hashes[name] = md5

        if md5 in md5_map:
            duplicates.append(name)
        else:
            md5_map[md5] = name

    unique_count = len(images) - len(duplicates)

    return {
        "duplicates": duplicates,
        "unique_count": unique_count,
        "hashes": hashes,
    }


def count_images_by_food(images: dict):

    counts = {}
    for name in images.keys():
        base = os.path.basename(name)

        # 1️⃣ Quitar extensión .png
        food = re.sub(r"\.png$", "", base, flags=re.IGNORECASE)

        # 2️⃣ Quitar la parte de tipo de conjunto (training, validation, evaluation)
        food = re.sub(r"-(training|validation|evaluation)", "", food, flags=re.IGNORECASE)

        # 3️⃣ Quitar números al final o intermedios (como -192 o -3321)
        food = re.sub(r"-?\d+$", "", food)

        # 4️⃣ Reemplazar guiones por espacios y limpiar
        food = food.replace("-", " ").strip()

        # 5️⃣ Normalizar capitalización
        food = food.capitalize()

        # 6️⃣ Contar
        counts[food] = counts.get(food, 0) + 1

    return counts


def process_images(client):
    keys: List[str] = list(list_objects(client, config.FORMATTED_BUCKET, config.FORMATTED_IMAGE_PATH))

    if not keys:
        print("[WARN] No formatted images found to validate.")
        return

    all_images = {}

    # First, we load the images
    with ProgressBar(
        total=len(keys),
        description="Loading images",
        unit="image",
        unit_scale=False,
    ) as progress:
        for key in keys:
            progress.set_description(f"Loading {os.path.basename(key)}", refresh=False)
            obj = client.get_object(config.FORMATTED_BUCKET, key)
            data = obj.read()
            obj.close()
            obj.release_conn()
            all_images[key] = data
            progress.update(1)

    # Validate images
    validated = validate_images(all_images)
    valid_images = validated["valid"]
    invalid_images = validated["invalid"]

    # Detect duplicates in the valid images
    duplicates_report = duplicates(valid_images)
    duplicate_names = set(duplicates_report["duplicates"])

    uploaded_trusted = 0
    uploaded_rejected = 0
    rejected_report = {}

    # Upload valid and non-duplicate images to trusted
    if valid_images:
        with ProgressBar(
            total=len(valid_images),
            description="Uploading trusted images",
            unit="image",
            unit_scale=False,
        ) as progress:
            for name, data in valid_images.items():
                progress.set_description(f"Trusted {os.path.basename(name)}", refresh=False)

                if name in duplicate_names:
                    progress.update(1)
                    continue

                dst_key = dst_key_for(name, config.TRUSTED_IMAGE_PATH)
                metadata = {
                    "x-amz-meta-source-key": name,
                    "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                    "x-amz-meta-format": "png",
                }
                client.put_object(
                    config.TRUSTED_BUCKET,
                    dst_key,
                    io.BytesIO(data),
                    length=len(data),
                    content_type="image/png",
                    metadata=metadata
                )
                uploaded_trusted += 1
                progress.update(1)

    if invalid_images:
        with ProgressBar(
            total=len(invalid_images),
            description="Uploading rejected images",
            unit="image",
            unit_scale=False,
        ) as progress:
            for name, reason in invalid_images.items():
                progress.set_description(f"Rejected {os.path.basename(name)}", refresh=False)
                data = all_images[name]
                dst_key = dst_key_for(name, config.REJECTED_IMAGE_PATH)
                metadata = {
                    "x-amz-meta-source-key": name,
                    "x-amz-meta-reason": reason,
                    "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                }
                client.put_object(
                    config.REJECTED_BUCKET,
                    dst_key,
                    io.BytesIO(data),
                    length=len(data),
                    content_type="image/png",
                    metadata=metadata
                )
                uploaded_rejected += 1

                # Keep it for the final report
                rejected_report.setdefault(reason, []).append(dst_key)
                progress.update(1)

        print("\nREJECTION REPORT")
        for reason, files in rejected_report.items():
            print(f" - {reason}: {len(files)} image(s) rejected")
    else:
        print("\nNo rejected images.")

    print(f"\nUploaded to Trusted: {uploaded_trusted}")
    print(f"Uploaded to Rejected: {uploaded_rejected}")

    # Report of the amount of types of images we have in the trusted bucket per group
    food_counts = count_images_by_food(valid_images)
    print("\n Validated images per food type:")
    if not food_counts:
        print("No valid images found.")
    else:
        for food, count in sorted(food_counts.items()):
            print(f" - {food}: {count}")


def main():
    try:
        client = get_minio_client()
        process_images(client)
    except S3Error as e:
        print(f"MinIO error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
