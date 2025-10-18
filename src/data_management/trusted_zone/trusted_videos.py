import os
import tempfile
import io
import subprocess
import json
import hashlib
import imageio_ffmpeg as ffmpeg
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from src.common.minio_client import get_minio_client
from minio.error import S3Error


# Buckets
FORMATTED_BUCKET = "formatted-zone"
TRUSTED_BUCKET = "trusted-zone"
REJECTED_BUCKET = "rejected-zone"

# Folders
SRC_PREFIX = "formatted/video_data/"
DST_PREFIX1 = "trusted/video_data/"
DST_PREFIX2 = "rejected/video_data/"

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

def is_video_valid(file_path: str):
    try:
        # just check if ffprobe can read the video file (no duration check)
        ffmpeg_path = ffmpeg.get_ffprobe_exe()
        subprocess.run(
            [
                ffmpeg_path, "-v", "error",
                "-show_entries", "format=format_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            stderr=subprocess.PIPE,  # capture errors if the video is corrupted
            text=True,               # decode output as text
            check=True               # if ffprobe fails then error
        )
        return True  # ffprobe could read it, so it's valid
    except subprocess.CalledProcessError:
        # ffprobe failed (file not readable or corrupted)
        return False

import shutil  # añade esto arriba del archivo

def video_properties(file_path: str):
    result = {
        "duration": None,
        "width": None,
        "height": None,
        "codec": None,
        "duration_ok": False,
        "resolution_ok": False,
        "codec_ok": False,
        "error": None
    }

    try:
        # 🔍 Intentar localizar ffprobe
        ffprobe_path = shutil.which("ffprobe")

        if not ffprobe_path:
            raise FileNotFoundError(
                "ffprobe not found. Please install ffmpeg and make sure it's in your PATH."
            )

        print(f"\n[DEBUG] Checking video: {os.path.basename(file_path)}")
        print(f"[DEBUG] Using ffprobe: {ffprobe_path}")

        cmd = [
            ffprobe_path, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name",
            "-show_entries", "format=duration",
            "-of", "json",
            file_path
        ]

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("[DEBUG CMD]", " ".join(cmd))
        print("[DEBUG RAW STDOUT]\n", process.stdout)
        print("[DEBUG RAW STDERR]\n", process.stderr)

        if not process.stdout.strip():
            raise RuntimeError("ffprobe did not return any data.")

        info = json.loads(process.stdout)

        duration = float(info.get("format", {}).get("duration", 0.0))
        width, height, codec = 0, 0, ""

        if "streams" in info and len(info["streams"]) > 0:
            first_stream = info["streams"][0]
            width = first_stream.get("width", 0)
            height = first_stream.get("height", 0)
            codec = first_stream.get("codec_name", "").lower()

        result.update({
            "duration": duration,
            "width": width,
            "height": height,
            "codec": codec,
            "duration_ok": duration >= 1.0,
            "resolution_ok": width >= 480 and height >= 360,
            "codec_ok": codec in ("h264", "hevc", "vp9", "avc1", "mpeg4"),
        })

        print(f"[INFO] duration={duration:.2f}s | resolution={width}x{height} | codec={codec}")
        print(f"[CHECKS] duration_ok={result['duration_ok']} | resolution_ok={result['resolution_ok']} | codec_ok={result['codec_ok']}")

    except FileNotFoundError as e:
        result["error"] = str(e)
        print("[ERROR]", e)
    except json.JSONDecodeError:
        result["error"] = "Invalid JSON output from ffprobe"
        print("[ERROR] Invalid JSON output from ffprobe")
    except Exception as e:
        result["error"] = str(e)
        print(f"[ERROR] while analyzing {file_path}: {e}")

    return result
def duplicates(videos: dict):
    md5_seen = {}
    duplicates = []

    for name, data in videos.items():
        md5 = hashlib.md5(data).hexdigest()
        if md5 in md5_seen:
            duplicates.append(name)  # store the duplicate name
        else:
            md5_seen[md5] = name

    # Return the set of duplicates
    return set(duplicates)


def process_videos(client):
    all_videos = {}
    for obj in client.list_objects(FORMATTED_BUCKET, prefix=SRC_PREFIX, recursive=True):
        if not obj.object_name.endswith("/"):
            file_obj = client.get_object(FORMATTED_BUCKET, obj.object_name)
            data = file_obj.read()
            file_obj.close()
            file_obj.release_conn()
            all_videos[obj.object_name] = data

    valid_videos = {}
    invalid_videos = {}

    # Validate videos
    for name, data in all_videos.items():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        props = video_properties(temp_path)
        os.remove(temp_path)

        if props["duration_ok"] and props["resolution_ok"] and props["codec_ok"]:
            valid_videos[name] = data
        else:
            reason = []
            if not props["duration_ok"]:
                reason.append("Invalid duration")
            if not props["resolution_ok"]:
                reason.append("Low resolution")
            if not props["codec_ok"]:
                reason.append("Unsupported codec")
            invalid_videos[name] = ", ".join(reason)

    # Detect duplicates among valid ones
    dupes = duplicates(valid_videos)

    uploaded_trusted = 0
    uploaded_rejected = 0

   
    for name, data in valid_videos.items():
        if name in dupes:
            continue

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        dst_key = dst_key_for(name, DST_PREFIX1)
        size = os.path.getsize(temp_path)
        metadata = {
            "x-amz-meta-source-key": name,
            "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
            "x-amz-meta-format": "mp4(h264)",
        }

        with open(temp_path, "rb") as f:
            client.put_object(
                TRUSTED_BUCKET,
                dst_key,
                data=f,
                length=size,
                content_type="video/mp4",
                metadata=metadata
            )

        uploaded_trusted += 1
        os.remove(temp_path)

    if invalid_videos:
        print("\n Uploading rejected videos")
        rejected_report = {}

        for name, reason in invalid_videos.items():
            data = all_videos[name]

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(data)
                temp_path = tmp.name

            dst_key = dst_key_for(name, DST_PREFIX2)
            size = os.path.getsize(temp_path)
            metadata = {
                "x-amz-meta-source-key": name,
                "x-amz-meta-reason": reason,
                "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
            }

            with open(temp_path, "rb") as f:
                client.put_object(
                    REJECTED_BUCKET,
                    dst_key,
                    data=f,
                    length=size,
                    content_type="video/mp4",
                    metadata=metadata
                )

            uploaded_rejected += 1
            os.remove(temp_path)

            rejected_report.setdefault(reason, []).append(dst_key)

        print("\nREJECTION REPORT")
        for reason, files in rejected_report.items():
            count = len(files)
            print(f" - {reason}: {count} video(s) rejected)")
    else:
        print("\n No rejected videos.")

    print(f"\nUploaded to Trusted: {uploaded_trusted}")
    print(f"Uploaded to Rejected: {uploaded_rejected}")
    


def main():
    try:
        client = get_minio_client()
        process_videos(client)
    except S3Error as e:
        print(f"MinIO error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()