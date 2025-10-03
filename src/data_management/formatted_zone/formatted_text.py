import os
import io
import json
from datetime import datetime, timezone 
from src.common.minio_client import get_minio_client
from minio.error import S3Error

# Buckets
LANDING_BUCKET = "landing-zone"
FORMATTED_BUCKET = "formatted-zone"

# Folders
SRC_PREFIX = "persistent_landing/text_data/"
DST_PREFIX = "formatted/text_data/"

FIELDS_TO_KEEP = ["title", "ingredients", "directions"]

def list_objects(client, bucket, prefix):
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name

def dst_key_for(src_key: str) -> str:
    if src_key.startswith(SRC_PREFIX):
        dst_key = src_key.replace(SRC_PREFIX, DST_PREFIX, 1)
    else:
        dst_key = DST_PREFIX + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".json"


def _has_content(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return bool(value)

def filter_entry(entry: dict, fields: list = None) -> dict:
    if fields is None:
        fields = FIELDS_TO_KEEP

    filtered = {}
    for field in fields:
        if field in entry:
            filtered[field] = entry[field]
    return filtered

def _normalize_root(data):
    if isinstance(data, dict) and "root" in data and isinstance(data["root"], list):
        return data["root"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON structure: expected list or dict with 'root'.")

def process_text(client, key: str):
    print("Processing:", key)

    # Download
    obj = client.get_object(LANDING_BUCKET, key)
    raw = obj.read()
    obj.close(); obj.release_conn()

    # Parse JSON
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except Exception as e:
        raise ValueError(f"Could not parse JSON in {key}: {e}")

    # Transform
    entries = _normalize_root(data)

    must_filtered = []
    for e in entries:
        if all((f in e) and _has_content(e[f]) for f in FIELDS_TO_KEEP):
            must_filtered.append(filter_entry(e))

    # Salida canónica y simple
    output = {
        "schema_version": 1,
        "root": must_filtered
    }

    # Upload
    payload = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    dst_key = dst_key_for(key)

    metadata = {
        "x-amz-meta-source-key": key,
        "x-amz-meta-processed-at": datetime.now(timezone.utc).isoformat(),
        "x-amz-meta-schema-version": "1",
    }

    client.put_object(
        FORMATTED_BUCKET,
        dst_key,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
        metadata=metadata
    )
    print(f"Saved in: {FORMATTED_BUCKET}/{dst_key}")

def main():
    client = get_minio_client()

    for key in list_objects(client, LANDING_BUCKET, SRC_PREFIX):
        try:
            process_text(client, key)
        except S3Error as e:
            print(f"MinIO error with {key}: {e}")
        except Exception as e:
            print(f"Error processing {key}: {e}")

if __name__ == "__main__":
    main()
