import os
import io
import re
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from src.common.minio_client import get_minio_client
from minio.error import S3Error
from PIL import Image, ImageStat, UnidentifiedImageError
#import cv2
import numpy as np
import hashlib
#from collections import defaultdict

# Buckets
FORMATTED_BUCKET = "formatted-zone"
TRUSTED_BUCKET = "trusted-zone"
REJECTED_BUCKET = "rejected-zone"
# Folders
SRC_PREFIX = "formatted/image_data/"
DST1_PREFIX = "trusted/image_data/"
DST2_PREFIX = "rejected/image_data/"

def list_objects(client, bucket, prefix):
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name

def dst_key_for(src_key: str, dst_prefix: str):
    if src_key.startswith(SRC_PREFIX):
        dst_key = src_key.replace(SRC_PREFIX, dst_prefix, 1)
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
        "error": None
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
    img = Image.open(io.BytesIO(data)).convert("L")  # grey's scale
    stat = ImageStat.Stat(img)# create an object with images features
    # we get the average from all the pixels  [0] and we standarize (from 0 to 1)
    brightness = stat.mean[0] / 255.0  
    contrast= stat.stddev[0] / 255.0  

    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "brightness_ok": 0.2 <= brightness <= 0.8,
        "contrast_ok": contrast >= 0.2
    }

# Function to combine above functions
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

def duplicates(images: dict):#check copy paste images (add in duplicates the name of the duplicated images)
    
    md5_map = {}        # hash -> nombre original si contiene el hash aqui dentro se añadira a duplicado
    duplicates = []     # nombres duplicados
    hashes = {}         # nombre -> hash de todas las imagenes

    for name, data in images.items():
        md5 = hashlib.md5(data).hexdigest()
        hashes[name] = md5

        if md5 in md5_map:
            # Ya existe este hash: duplicado
            duplicates.append(name)
        else:
            # Primera vez que aparece este hash: conservar
            md5_map[md5] = name

    unique_count = len(images) - len(duplicates)

    return {
        "duplicates": duplicates, 
        "unique_count": unique_count,
        "hashes": hashes           
    }

def count_images_by_food(images: dict):

    counts = {}
    for name in images.keys():
        base = os.path.basename(name)
        food = re.sub(r"-(training|validation|evaluation)-", "-", base, flags=re.IGNORECASE)# we ignore what type belong
        food = re.sub(r"\.png$", "", food, flags=re.IGNORECASE)
        food = food.replace("-", " ").strip().capitalize()

        counts[food] = counts.get(food, 0) + 1

    return counts
def process_images(client):

    all_images = {}
    for key in list_objects(client, FORMATTED_BUCKET, SRC_PREFIX):
        obj = client.get_object(FORMATTED_BUCKET, key)
        data = obj.read()
        obj.close()
        obj.release_conn()
        all_images[key] = data

    # Validate images
    validated = validate_images(all_images)
    valid_images = validated["valid"]
    invalid_images = validated["invalid"]

    # Detect duplicates in the valid images
    duplicates_report = duplicates(valid_images)
    duplicate_names = set(duplicates_report["duplicates"])
    hashes = duplicates_report["hashes"]

    uploaded_trusted = 0
    uploaded_rejected = 0
    rejected_report = {}

    # Upload valid and non-duplicate images to trusted
    if valid_images:
        print("\n Uploading valid, non-duplicate images to Trusted Zone")
        for name, data in valid_images.items():
            if name in duplicate_names:
                # if it is duplicated to Rejected Zone
                print(f" Duplicate detected, moving to rejected: {name}")
                dst_key = dst_key_for(name, DST2_PREFIX)
                metadata = {
                    "x-amz-meta-source-key": name,
                    "x-amz-meta-reason": "duplicate image",
                    "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                }
                client.put_object(
                    REJECTED_BUCKET,
                    dst_key,
                    io.BytesIO(data),
                    length=len(data),
                    content_type="image/png",
                    metadata=metadata
                )
                uploaded_rejected += 1
                rejected_report.setdefault("duplicate image", []).append(dst_key)
                continue

            # Image valid and unique
            dst_key = dst_key_for(name, DST1_PREFIX)
            metadata = {
                "x-amz-meta-source-key": name,
                "x-amz-meta-hash": hashes[name],
                "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                "x-amz-meta-format": "png",
            }

            client.put_object(
                TRUSTED_BUCKET,
                dst_key,
                io.BytesIO(data),
                length=len(data),
                content_type="image/png",
                metadata=metadata
            )
            uploaded_trusted += 1
            print(f" Uploaded to Trusted: {dst_key}")

    
    if invalid_images:
        print("\n Uploading invalid images to Rejected Zone")
        for name, reason in invalid_images.items():
            data = all_images[name]
            dst_key = dst_key_for(name, DST2_PREFIX)
            metadata = {
                "x-amz-meta-source-key": name,
                "x-amz-meta-reason": reason,
                "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
            }
            client.put_object(
                REJECTED_BUCKET,
                dst_key,
                io.BytesIO(data),
                length=len(data),
                content_type="image/png",
                metadata=metadata
            )
            uploaded_rejected += 1
            print(f" Uploaded to Rejected: {dst_key}")

    # Summary
    if rejected_report:
        print("\n REJECTION REPORT:")
        for reason, files in rejected_report.items():
            print(f" - {reason}: {len(files)} image(s) rejected")
    else:
        print("\n No rejected images.")

    print(f"\n Uploaded to Trusted: {uploaded_trusted}")
    print(f" Uploaded to Rejected: {uploaded_rejected}")

    # Count of images
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