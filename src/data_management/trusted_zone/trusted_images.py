import os
import io
import re
import hashlib
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo
from minio.error import S3Error
from PIL import Image, ImageStat, UnidentifiedImageError

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


def list_objects(client, bucket, prefix):
    """List all objects from a given MinIO bucket and prefix."""
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name


def dst_key_for(src_key: str, dst_prefix: str):
    """Generate the destination key (path) for the trusted text."""
    if src_key.startswith(config.FORMATTED_IMAGE_PATH):
        dst_key = src_key.replace(config.FORMATTED_IMAGE_PATH, dst_prefix, 1)
    else:
        dst_key = os.path.join(dst_prefix, os.path.basename(src_key))
    return dst_key

"""Functions to validate image integrity and quality by checking format, size, color mode, brightness, and contrast:"""

def is_image_valid(data: bytes):
    """Check if the image data is valid and not corrupted."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def image_properties(data: bytes, expected_size=(512, 512)):
    """Check if the image meets expected format, size, and color mode requirements."""
    result = {
        "format": None,
        "size": None,
        "color": None,
        "size_ok": False,
        "color_ok": False,
        "format_ok": False,
        "error": None,
    }
    
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()

            result["format"] = img.format
            result["size"] = img.size
            result["color"] = img.mode

            if img.size == expected_size:   # Check if image dimensions match expected size (512x512).
                result["size_ok"] = True

            if img.mode in ("RGB", "RGBA"): # Accept only RGB or RGBA color modes.
                result["color_ok"] = True

            if img.format == "PNG":         # Accept only PNG image format.       
                result["format_ok"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def brightness_and_contrast(data: bytes):
    """Calculate image brightness and contrast, and check if they are within acceptable ranges."""
    img = Image.open(io.BytesIO(data)).convert("L")  # Convert image to grayscale.
    stat = ImageStat.Stat(img)                       # Compute image statistics.
    brightness = stat.mean[0] / 255.0                # Normalize average brightness to [0, 1].
    contrast = stat.stddev[0] / 255.0                # Normalize contrast (standard deviation) to [0, 1].

    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "brightness_ok": 0.2 <= brightness <= 0.8,
        "contrast_ok": contrast >= 0.2,
    }

def validate_images(images: dict):
    """Apply the above functions to check if the image is valid or not and store it in a valid or invalid images dictionary"""
    valid_images = {}
    invalid_images = {}


    for name, data in images.items():
        if not is_image_valid(data):
            invalid_images[name] = "Corrupt or unreadable"  # check if it's readable
            continue

        props = image_properties(data)                      # size, format and color
        bright = brightness_and_contrast(data)              # brightness and contrast

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
    """Identify duplicate images by comparing their MD5 hashes."""
    md5_seen = {}   # Store MD5 hash of the images.
    duplicates = [] # Store names of the duplicate images.
     
    for name, data in images.items():
        md5 = hashlib.md5(data).hexdigest()
        if md5 in md5_seen:
            duplicates.append(name)
        else:
            md5_seen[md5] = name

    return set(duplicates)


def count_images_by_food(images: dict):
    """Count how many images exist for each food category based on image filenames."""
    counts = {}
    for name in images.keys():
        base = os.path.basename(name)
        food = re.sub(r"\.png$", "", base, flags=re.IGNORECASE) # Remove from the image name the png extension.
        food = re.sub(r"-(training|validation|evaluation)", "", food, flags=re.IGNORECASE) # Remove the group where they belong.
        food = re.sub(r"-?\d+$", "", food)                      # Remove the number.
        food = food.replace("-", " ").strip()                   # Remove begin and end spaces.
        food = food.capitalize()                                # Capitalize the name.
        counts[food] = counts.get(food, 0) + 1                  # Increase the image counter.

    return counts

"""Function to apply the validation process to all images from the formatted zone, saving the output in the trusted zone or in the rejected zone:"""

def process_images(client):
    keys: List[str] = list(list_objects(client, config.FORMATTED_BUCKET, config.FORMATTED_IMAGE_PATH))

    if not keys:
        print("[WARN] No formatted images found to validate.")
        return

    all_images = {}

    # Download all images and store them in the all_images dictionary showing the progress.
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

    # Split images into valid and invalid sets, and identify duplicates in the valid set.
    validated = validate_images(all_images)
    valid_images = validated["valid"]
    invalid_images = validated["invalid"]

    duplicate_names = duplicates(valid_images)
    
    uploaded_trusted = 0
    uploaded_rejected = 0
    rejected_report = {}

    # Upload valid and non-duplicate images to trusted with the metadata, showing the progress.
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

    # Upload invalid images with the metadata to the rejected bucket showing the progress.
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

                rejected_report.setdefault(reason, []).append(dst_key)
                progress.update(1)

        # Summary of the amount of images stored in the rejected bucket and the reason.
        print("\nREJECTION REPORT")
        for reason, files in rejected_report.items():
            print(f" - {reason}: {len(files)} image(s) rejected")
    else:
        print("\nNo rejected images.")

    # Amount of images in each zone.
    print(f"\nUploaded to Trusted: {uploaded_trusted}")
    print(f"Uploaded to Rejected: {uploaded_rejected}")

    # Report of the amount of food types of images there are in the trusted bucket per group.
    food_counts = count_images_by_food(valid_images)
    print("\n Validated images per food type:")
    if not food_counts:
        print("No valid images found.")
    else:
        for food, count in sorted(food_counts.items()):
            print(f" - {food}: {count}")


def main():
    """Main entry point: process all images, apply validation rules, and display progress."""
    try:
        client = get_minio_client()
        process_images(client)
    except S3Error as e:
        print(f"MinIO error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
