from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error

from src.common.minio_client import get_minio_client


LANDING_BUCKET = "landing-zone"
TEMPORAL_PREFIX = "temporal_landing/"
PERSISTENT_PREFIX = "persistent_landing/"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
}

TEXT_EXTENSIONS = {
    ".json",
    ".txt",
}


@dataclass
class ObjectMove:
    source: str
    destination: str


def classify_destination(object_name: str):
    """Return the persistent prefix that matches the object's extension."""
    suffix = object_name.lower().rsplit(".", 1)
    extension = f".{suffix[-1]}" if len(suffix) == 2 else ""

    if extension in IMAGE_EXTENSIONS:
        return f"{PERSISTENT_PREFIX}image_data/"

    if extension in VIDEO_EXTENSIONS:
        return f"{PERSISTENT_PREFIX}video_data/"

    if extension in TEXT_EXTENSIONS:
        return f"{PERSISTENT_PREFIX}text_data/"

    return None


# Iterate objects in temporal bucket and classify them to move them to each persistent folder.
# This function is a generator, meaning that it delivers ObjectMove one by one to whoever calls it (main function).
def iter_objects_in_temporal_bucket(client: Minio):
    for obj in client.list_objects(LANDING_BUCKET, prefix=TEMPORAL_PREFIX, recursive=True):
        name = obj.object_name

        if obj.is_dir or not name or name == TEMPORAL_PREFIX:
            continue

        target_prefix = classify_destination(name)
        if not target_prefix:
            print(f"[WARN] Unknown type, skipping: {name}")
            continue

        relative_path = name.removeprefix(TEMPORAL_PREFIX)
        if not relative_path:
            print(f"[WARN] Missing relative name, skipping: {name}")
            continue

        destination = f"{target_prefix}{relative_path}"
        yield ObjectMove(source=name, destination=destination)


# Copy the object to the persistent path and delete it from temporal landing.
def move_object(client: Minio, move: ObjectMove):
    copy_source = CopySource(LANDING_BUCKET, move.source)
    client.copy_object(LANDING_BUCKET, move.destination, copy_source)
    client.remove_object(LANDING_BUCKET, move.source)
    print(f"[OK] {move.source} -> {move.destination}")


# Process every temporal object and move it into the matching persistent folder.
def main():
    client = get_minio_client()

    moved = 0
    for move in iter_objects_in_temporal_bucket(client):
        move_object(client, move)
        moved += 1

    if moved == 0:
        print("[INFO] No objects found in temporal_landing to move.")
    else:
        print(f"[OK] Objects moved: {moved}")


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print(f"MinIO error: {exc}")
