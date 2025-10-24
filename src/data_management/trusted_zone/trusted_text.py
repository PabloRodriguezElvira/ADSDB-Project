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
    """List all objects from a given MinIO bucket and prefix."""
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name


def dst_key_for(src_key: str):
    """Generate the destination key (path) for the trusted text."""
    if src_key.startswith(config.FORMATTED_TEXT_PATH):
        dst_key = src_key.replace(config.FORMATTED_TEXT_PATH, config.TRUSTED_TEXT_PATH, 1)
    else:
        dst_key = config.TRUSTED_TEXT_PATH + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".json"

"""Functions to clean, anonymize, and lowercase text:"""

def remove_whitespace(text: str):
    """Remove extra whitespace and invisible characters."""
    if isinstance(text, str):
        text = text.replace('\u200b', '') # Remove zero-width space characters (invisible, sometimes added by text editors).
        text = text.replace('\ufeff', '') # Remove BOM (Byte Order Mark) that can appear at start of UTF-8 files.
        text = re.sub(r'\s+', ' ', text)  # Replace any sequence of whitespace (spaces, tabs, newlines) with a single space.
        return text.strip()               # Remove first and last spaces.
    else:
        return text

def anonymize_text(text: str):
    """Replace personal information (emails, URLs, users, phones) with placeholder tags."""
    if isinstance(text, str):
      text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
      text = re.sub(r'(https?://\S+|www\.\S+)', '[URL]', text)
      text = re.sub(r'@\w+', '[USER]', text)
      text = re.sub(r'\+?\d[\d\s\-]{7,}\d', '[PHONE]', text)
      return text
    else:
      return text

def to_lower_preserving_tags(text: str):
    """Convert text to lowercase while preserving tags."""
    if isinstance(text, str):
      tags = re.compile(r'\[([A-Z]+)\]')
      lowered = text.lower() 
      return tags.sub(lambda m: f'[{m.group(1)}]', lowered) # Return the tags to the lowercase text.
    else:
      return text

def clean_text(text: str):
    """"Clean text by combining the three preprocessing functions."""
    text = remove_whitespace(text)
    text = anonymize_text(text)
    text = to_lower_preserving_tags(text)
    return text

"""Function to apply a clean process to a JSON file from the formatted zone, saving the output in the trusted zone:"""

def process_text(client, key: str, progress: Optional[ProgressBar] = None):
    if progress:
        progress.set_description(f"Processing {os.path.basename(key)}", refresh=False)
        
    # Download the JSON file.
    obj = client.get_object(config.FORMATTED_BUCKET, key)
    raw = obj.read()
    obj.close(); obj.release_conn()

    # Decode and parse the JSON file, then extract and validate the 'root' list of text entries.
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Could not parse JSON in {key}: {e}")

    entries = data.get("root", [])

    # Ensure that 'root' exists and contains a valid list of entries.
    if not isinstance(entries, list) or not entries:
        print(f"No valid 'root' list found in {key}")
        return

    # Apply to every recipe the clean_text function.
    cleaned_entries = []
    for entry in entries:
        if isinstance(entry, str):          
            cleaned_entry = clean_text(entry)  
            cleaned_entries.append(cleaned_entry) 

    # Generate the output as in the formatted-zone.
    output = {
        "schema_version": 1,
        "root": cleaned_entries
    }

    # Serialize the cleaned data back to JSON, define metadata, and upload it to the trusted zone.
    transform = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    dst_key = dst_key_for(key)

    metadata = {
        "x-amz-meta-source-key": key,
        "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        "x-amz-meta-schema-version": "1",
    }

    
    client.put_object(
        config.TRUSTED_BUCKET,
        dst_key,
        io.BytesIO(transform),
        length=len(transform),
        content_type="application/json",
        metadata=metadata
    )

def main():
    """Main entry point: process all formatted JSON files, apply text cleaning, and show progress."""
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
    