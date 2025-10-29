import os
import tempfile
import subprocess
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Iterable, List
from zoneinfo import ZoneInfo
from minio.error import S3Error
import re, json, subprocess, imageio_ffmpeg as iio



from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


def list_objects(client, bucket, prefix) -> Iterable[str]:
    """List all objects from a given MinIO bucket and prefix."""
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name

def dst_key_for(src_key: str, dst_prefix: str):
    """Generate the destination key (path) for the trusted text."""
    if src_key.startswith(config.FORMATTED_VIDEO_PATH):
        dst_key = src_key.replace(config.FORMATTED_VIDEO_PATH, dst_prefix, 1)
    else:
        dst_key = os.path.join(dst_prefix, os.path.basename(src_key))
    return dst_key


def video_properties(path: str):
    """Extract duration, resolution, and codec info from a video using ffprobe and validates them."""

    DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
    STREAM_RE = re.compile(r"Stream #\d+:\d+.*Video:\s*([^,]+).*?(\d+)x(\d+)")

    res = {"duration": None, "width": None, "height": None, "codec": None,
           "duration_ok": False, "resolution_ok": False, "codec_ok": False, "error": None}
    try:
        ffmpeg = iio.get_ffmpeg_exe()  # ffmpeg from imageio-ffmpeg
        
        # Call ffmpeg to get the properties.
        p = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", check=False
        )
        out = p.stderr  # ffmpeg prints metadata in stderr

        # Extract video duration from the ffmpeg result
        m = DUR_RE.search(out)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            duration = h*3600 + mi*60 + s
            res["duration"] = duration
            res["duration_ok"] = duration > 1.0

        # Extract video codec, width and height from the ffmpeg result
        m = STREAM_RE.search(out)
        if m:
            codec = m.group(1).lower().strip()
            codec_clean = codec.lower().split()[0]       
            codec_clean = codec_clean.replace("(", "").replace(")", "")
            w, h = int(m.group(2)), int(m.group(3))

            res.update({"codec": codec, "width": w, "height": h})
            res["resolution_ok"] = (w >= 480 and h >= 360)
            res["codec_ok"] = codec_clean in ("h264", "hevc", "vp9")

    except Exception as e:
        res["error"] = str(e)
    return res

def duplicates(videos: dict):
    """Identify duplicate videos by comparing their MD5 hashes."""
    md5_seen = {}      # Store MD5 hash of the images.
    duplicates = []    # Store names of the duplicate images.

    for name, data in videos.items():
        md5 = hashlib.md5(data).hexdigest()
        if md5 in md5_seen:
            duplicates.append(name)  
        else:
            md5_seen[md5] = name

    return set(duplicates)

"""Function to apply the validation process to all videos from the formatted zone, saving the output in the trusted zone or in the rejected zone:"""

def process_videos(client):
    keys: List[str] = list(list_objects(client, config.FORMATTED_BUCKET, config.FORMATTED_VIDEO_PATH))

    if not keys:
        print("[WARN] No formatted videos found to validate.")
        return

    all_videos = {}

    # Download all images and store them in the all_videos dictionary showing the progress.
    with ProgressBar(
        total=len(keys),
        description="Loading videos",
        unit="file",
        unit_scale=False,
    ) as progress:
        for key in keys:
            progress.set_description(f"Loading {Path(key).name}", refresh=False)
            file_obj = client.get_object(config.FORMATTED_BUCKET, key)
            data = file_obj.read()
            file_obj.close()
            file_obj.release_conn()
            all_videos[key] = data
            progress.update(1)


    # Split images into valid and invalid sets showing the progress.
    valid_videos = {}
    invalid_videos = {}
    
    with ProgressBar(
        total=len(all_videos),
        description="Validating videos",
        unit="file",
        unit_scale=False,
    ) as progress:
        for name, data in all_videos.items(): # Run through all videos, checking the properties of each one
            progress.set_description(f"Validating {Path(name).name}", refresh=False)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(data)
                temp_path = tmp.name

            props = video_properties(temp_path)
            print(props)
            os.remove(temp_path)

            if props["duration_ok"] and props["resolution_ok"] and props["codec_ok"]:
                valid_videos[name] = data
            else:
                reasons = []
                if not props["duration_ok"]:
                    reasons.append("Invalid duration")
                if not props["resolution_ok"]:
                    reasons.append("Low resolution")
                if not props["codec_ok"]:
                    reasons.append("Unsupported codec")
                invalid_videos[name] = ", ".join(reasons)
            progress.update(1)

    dupes = duplicates(valid_videos) # Check duplicates in the valid_videos set.

    
    uploaded_trusted = 0
    uploaded_rejected = 0

    # Upload the valid and non-duplicate videos with the metadata in the trusted bucket, showing the progress.
    if valid_videos:
        with ProgressBar(
            total=len(valid_videos),
            description="Uploading trusted videos",
            unit="file",
            unit_scale=False,
        ) as progress:
            for name, data in valid_videos.items():
                progress.set_description(f"Trusted {Path(name).name}", refresh=False)
                if name in dupes:
                    progress.update(1)
                    continue

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(data)
                    temp_path = tmp.name

                dst_key = dst_key_for(name, config.TRUSTED_VIDEO_PATH)
                size = os.path.getsize(temp_path)
                metadata = {
                    "x-amz-meta-source-key": name,
                    "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                    "x-amz-meta-format": "mp4(h264/acc)",
                }

                with open(temp_path, "rb") as f:
                    client.put_object(
                        config.TRUSTED_BUCKET,
                        dst_key,
                        data=f,
                        length=size,
                        content_type="video/mp4",
                        metadata=metadata,
                    )

                uploaded_trusted += 1
                os.remove(temp_path)
                progress.update(1)

    # Upload the invalid videos with the metadata to the rejected bucket, showing the progress.
    if invalid_videos:
        rejected_report = {}
        with ProgressBar(
            total=len(invalid_videos),
            description="Uploading rejected videos",
            unit="file",
            unit_scale=False,
        ) as progress:
            for name, reason in invalid_videos.items():
                progress.set_description(f"Rejected {Path(name).name}", refresh=False)
                data = all_videos[name]

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(data)
                    temp_path = tmp.name

                dst_key = dst_key_for(name, config.REJECTED_VIDEO_PATH)
                size = os.path.getsize(temp_path)
                metadata = {
                    "x-amz-meta-source-key": name,
                    "x-amz-meta-reason": reason,
                    "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
                }

                with open(temp_path, "rb") as f:
                    client.put_object(
                        config.REJECTED_BUCKET,
                        dst_key,
                        data=f,
                        length=size,
                        content_type="video/mp4",
                        metadata=metadata,
                    )

                uploaded_rejected += 1
                os.remove(temp_path)

                rejected_report.setdefault(reason, []).append(dst_key)
                progress.update(1)
        # Summary of the amount of videos stored in the rejected bucket and the reason.
        print("\nREJECTION REPORT")
        for reason, files in rejected_report.items():
            count = len(files)
            print(f" - {reason}: {count} video(s) rejected")
    else:
        print("\nNo rejected videos.")

    # Amount of videos in each zone.
    print(f"\nUploaded to Trusted: {uploaded_trusted}")
    print(f"Uploaded to Rejected: {uploaded_rejected}")

def main():
    """Main entry point: process all videos, apply validation rules, and display progress."""
    try:
        client = get_minio_client()
        process_videos(client)
    except S3Error as e:
        print(f"MinIO error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
