import os
import io
import json
import re
from typing import List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


FIELDS_TO_KEEP = ["title", "ingredients", "directions"]

def list_objects(client, bucket, prefix):
    """List all object names in a given S3 bucket and prefix, skipping folders."""
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name


def dst_key_for(src_key: str):
    """Generate the destination key (path) for the processed JSON file."""
    if src_key.startswith(config.LANDING_TEXT_PATH):
        dst_key = src_key.replace(config.LANDING_TEXT_PATH, config.FORMATTED_TEXT_PATH, 1)
    else:
        dst_key = config.FORMATTED_TEXT_PATH + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".json"


def _has_content(value):
    """Check if a value is not empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple)):
        return any(_has_content(v) for v in value)
    return bool(value)


def _normalize_root(data):
    """Normalize JSON structure: ensure we return a list of entries."""
    if isinstance(data, dict) and "root" in data and isinstance(data["root"], list):
        return data["root"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure: expected list or dict with 'root'.")


def _to_space_text(value):
    """Convert any value to a clean single-line string."""
    if value is None:
        s = ""
    elif isinstance(value, str):
        s = value
    elif isinstance(value, (list, tuple)):
        s = " ".join(str(v) for v in value if str(v).strip())
    else:
        s = str(value)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def process_text(client, key: str, progress: Optional[ProgressBar] = None):
    """Download, process, and upload a cleaned JSON text file."""
    if progress:
        progress.set_description(f"Processing {os.path.basename(key)}", refresh=False)
    else:
        print("Processing:", key)

    # Download JSON file from MinIO
    obj = client.get_object(config.LANDING_BUCKET, key)
    raw = obj.read()
    obj.close(); obj.release_conn()

    # Parse JSON content
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except Exception as e:
        raise ValueError(f"Could not parse JSON in {key}: {e}")

    # Extract and validate entries
    entries = _normalize_root(data)

    flattened = []
    for e in entries:
        # Keep entries with all required fields and valid content
        if all((f in e) and _has_content(e[f]) for f in FIELDS_TO_KEEP):
            title = _to_space_text(e.get("title"))
            ingredients = _to_space_text(e.get("ingredients"))
            directions = _to_space_text(e.get("directions"))

            # Join fields into a single text block
            joined = " ".join(x for x in [title, ingredients, directions] if x)
            if joined:
                flattened.append(joined)

    if not flattened:
        raise ValueError(f"No valid entries with required fields in {key}")

    # Build final JSON structure
    output = {
        "root": flattened
    }

    # Upload processed JSON to destination bucket
    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    dst_key = dst_key_for(key)

    metadata = {
        "x-amz-meta-source-key": key,
        "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        "x-amz-meta-output-format": "json_flat_concat"
    }

    client.put_object(
        config.FORMATTED_BUCKET,
        dst_key,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
        metadata=metadata
    )


def main():
    """Main entry point: list files, process each, and show progress."""
    client = get_minio_client()

    # Get all text files to process
    keys: List[str] = list(list_objects(client, config.LANDING_BUCKET, config.LANDING_TEXT_PATH))

    if not keys:
        print("[WARN] No text files found to process.")
        return

    # Process files with a progress bar
    with ProgressBar(
        total=len(keys),
        description="Processing texts",
        unit="file",
        unit_scale=False,
    ) as progress:
        for key in keys:
            try:
                process_text(client, key, progress=progress)
            except S3Error as e:
                progress.write(f"MinIO error with {key}: {e}")
            except Exception as e:
                progress.write(f"Error processing {key}: {e}")
            finally:
                progress.update(1)


if __name__ == "__main__":
    main()
