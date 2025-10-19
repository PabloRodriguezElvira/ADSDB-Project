from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error

from src.common.minio_client import get_minio_client
from src.common.progress_bar import ProgressBar


LANDING_BUCKET = "landing-zone"
TEMPORAL_PREFIX = "temporal_landing/"
PERSISTENT_PREFIX = "persistent_landing/"


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}
TEXT_EXTENSIONS = {".json", ".txt"}


# Data class to represent a move operation
@dataclass
class ObjectMove:
    source: str
    destination: str


def classify_destination(object_name: str):
    """
    Return the persistent prefix matching the object's extension.
    """

    suffix = object_name.lower().rsplit(".", 1)
    extension = f".{suffix[-1]}" if len(suffix) == 2 else ""

    if extension in IMAGE_EXTENSIONS:
        return f"{PERSISTENT_PREFIX}image_data/"
    if extension in VIDEO_EXTENSIONS:
        return f"{PERSISTENT_PREFIX}video_data/"
    if extension in TEXT_EXTENSIONS:
        return f"{PERSISTENT_PREFIX}text_data/"
    return None


def iter_objects_in_temporal_bucket(client: Minio):
    """
    Generator: iterate temporal objects and yield move operations
    """

    for obj in client.list_objects(LANDING_BUCKET, prefix=TEMPORAL_PREFIX, recursive=True):
        name = obj.object_name

        # Skip invalid or directory entries
        if obj.is_dir or not name or name == TEMPORAL_PREFIX:
            continue

        target_prefix = classify_destination(name)
        if not target_prefix:
            print(f"[WARN] Unknown type, skipping: {name}")
            continue

        # Compute destination path
        relative_path = name.removeprefix(TEMPORAL_PREFIX)
        if not relative_path:
            print(f"[WARN] Missing relative name, skipping: {name}")
            continue

        destination = f"{target_prefix}{relative_path}"
        yield ObjectMove(source=name, destination=destination)


def move_object(client: Minio, move: ObjectMove):
    """
    Copy the object to persistent folder and remove the temporal one
    """

    copy_source = CopySource(LANDING_BUCKET, move.source)
    client.copy_object(LANDING_BUCKET, move.destination, copy_source)
    client.remove_object(LANDING_BUCKET, move.source)


def main():
    """
    Main process: move all temporal objects to their persistent destinations
    """

    client = get_minio_client()
    objects_to_move = list(iter_objects_in_temporal_bucket(client))

    if not objects_to_move:
        print("[INFO] No objects found in temporal_landing to move.")
        return

    with ProgressBar(
        total=len(objects_to_move),
        description="Moving landing objects",
        unit="file",
        unit_scale=False,
        unit_divisor=1,
    ) as progress:
        for obj in objects_to_move:
            progress.set_description(f"Moving {obj.source}", refresh=True)
            move_object(client, obj)
            progress.update(1)

        progress.write(f"[OK] Objects moved: {len(objects_to_move)}")



if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print(f"MinIO error: {exc}")
