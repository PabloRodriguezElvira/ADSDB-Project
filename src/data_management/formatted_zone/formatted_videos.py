import os
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo
import imageio_ffmpeg as ff
from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar


# Buckets
LANDING_BUCKET = "landing-zone"
FORMATTED_BUCKET = "formatted-zone"

# Folders
SRC_PREFIX = "persistent_landing/video_data/"
DST_PREFIX = "formatted/video_data/"

def list_objects(client, bucket, prefix) -> Iterable[str]:
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        if not obj.object_name.endswith("/"):
            yield obj.object_name
            
def transcode_to_mp4(in_path, out_path):
    ffmpeg_bin = ff.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_path
    ]
    subprocess.run(cmd, check=True)

def dst_key_for(src_key: str) -> str:
    if src_key.startswith(SRC_PREFIX):
        dst_key = src_key.replace(SRC_PREFIX, DST_PREFIX, 1)
    else:
        dst_key = DST_PREFIX + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".mp4"


def process_video(client, key: str, progress: Optional[ProgressBar] = None):

    ext = Path(key).suffix.lower()

    if progress:
        progress.set_description(f"Processing {Path(key).name}", refresh=False)
    else:
        print("Processing:", key)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in" + ext)
        out_path = os.path.join(tmp, "out.mp4")

        # Download
        obj = client.get_object(LANDING_BUCKET, key)
        with open(in_path, "wb") as f:
            f.write(obj.read())
        obj.close(); obj.release_conn()

        # Convert
        transcode_to_mp4(in_path, out_path)

        # Upload
        dst_key = dst_key_for(key)
        size = os.path.getsize(out_path)
        metadata = {
            "x-amz-meta-source-key": key,
            "x-amz-meta-processed-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
            "x-amz-meta-format": "mp4(h264/aac)"
        }
        with open(out_path, "rb") as f:
            client.put_object(
                FORMATTED_BUCKET,
                dst_key,
                data=f,
                length=size,
                content_type="video/mp4",
                metadata = metadata
            )


def main():
    client = get_minio_client()
    keys: List[str] = list(list_objects(client, config.LANDING_BUCKET, config.LANDING_VIDEO_PATH))

    if not keys:
        print("[WARN] No videos found to process.")
        return

    with ProgressBar(
        total=len(keys),
        description="Processing videos",
        unit="file",
        unit_scale=False,
    ) as progress:
        for key in keys:
            try:
                process_video(client, key, progress=progress)
            except S3Error as e:
                progress.write(f"MinIO error with {key}: {e}")
            except subprocess.CalledProcessError as e:
                progress.write(f"ffmpeg error with {key}: {e}")
            except Exception as e:
                progress.write(f"Unexpected error with {key}: {e}")
            finally:
                progress.update(1)

if __name__ == "__main__":
    main()
