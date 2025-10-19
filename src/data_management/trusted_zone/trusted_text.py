
import os
import io
import json
import re
from datetime import datetime
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo
from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


def list_objects(client, bucket, prefix) -> Iterable[str]:
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name


def dst_key_for(src_key: str):
    if src_key.startswith(config.FORMATTED_TEXT_PATH):
        dst_key = src_key.replace(config.FORMATTED_TEXT_PATH, config.TRUSTED_TEXT_PATH, 1)
    else:
        dst_key = config.TRUSTED_TEXT_PATH + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".json"

# functions to clean
# remove when we have more than one white spaces
def remove_whitespace(text: str):
    if isinstance(text, str):
      text = text.replace('\u200b', '')   # zero-width space
      text = text.replace('\ufeff', '')   # BOM mark (invisible al inicio)
      text = re.sub(r'\s+', ' ', text)    # replaces any sequence of spaces, tabs, or newlines with a single space
      return text.strip() # removes the first and last space
    else:
      return text

# function to anonymize
def anonymize_text(text: str):
    if isinstance(text, str):
      text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
      text = re.sub(r'(https?://\S+|www\.\S+)', '[URL]', text)
      text = re.sub(r'@\w+', '[USER]', text)
      text = re.sub(r'\+?\d[\d\s\-]{7,}\d', '[PHONE]', text)
      return text
    else:
      return text

# lower words but preserving the tags
def to_lower_preserving_tags(text: str):
    if isinstance(text, str):
      tags = re.compile(r'\[([A-Z]+)\]')
      lowered = text.lower()
      return tags.sub(lambda m: f'[{m.group(1)}]', lowered)# return the pattern
    else:
      return text

# combining the three functions
def clean_text(text: str):
    text = remove_whitespace(text)
    text = anonymize_text(text)
    text = to_lower_preserving_tags(text)
    return text

def process_text(client, key: str, progress: Optional[ProgressBar] = None):
    if progress:
        progress.set_description(f"Processing {os.path.basename(key)}", refresh=False)

    obj = client.get_object(config.FORMATTED_BUCKET, key)
    raw = obj.read()
    obj.close(); obj.release_conn()

    # transforming the json in a python
    try:
        data = json.loads(raw.decode("utf-8"))# we convert it to a python object to apply functions
    except Exception as e:
        raise ValueError(f"Could not parse JSON in {key}: {e}")

    entries = data.get("root", [])

    if not isinstance(entries, list) or not entries:# check if we really got a list ["...","..."]
        print(f"No valid 'root' list found in {key}")
        return

    # Clean function to every recipe 
    cleaned_entries = []
    for entry in entries:
        if isinstance(entry, str):          
            cleaned_entry = clean_text(entry)  # clean function to every "recipe"
            cleaned_entries.append(cleaned_entry)  # keep it in the list

    # Generate the output as in the formatted-zone
    output = {
        "schema_version": 1,
        "root": cleaned_entries
    }

    # we convert the python in a json
    transform = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    dst_key = dst_key_for(key)

    metadata = {
        "x-amz-meta-source-key": key,
        "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        "x-amz-meta-schema-version": "1",
    }

    # Uploaded
    client.put_object(
        config.TRUSTED_BUCKET,
        dst_key,
        io.BytesIO(transform),
        length=len(transform),
        content_type="application/json",
        metadata=metadata
    )

def main():
    client = get_minio_client()

    keys: List[str] = list(list_objects(client, config.FORMATTED_BUCKET, config.FORMATTED_TEXT_PATH))

    if not keys:
        print("[WARN] No formatted texts found to process.")
        return

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
