import os
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import imageio_ffmpeg as ff

from src.common.minio_client import get_minio_client
from minio.error import S3Error
import src.common.global_variables as config


def list_objects(client, bucket, prefix):
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
    if src_key.startswith(config.LANDING_VIDEO_PATH):
        dst_key = src_key.replace(config.LANDING_VIDEO_PATH, config.FORMATTED_VIDEO_PATH, 1)
    else:
        dst_key = config.FORMATTED_VIDEO_PATH + os.path.basename(src_key)
    base, _ = os.path.splitext(dst_key)
    return base + ".mp4"

def process_video(client, key: str):
    ext = Path(key).suffix.lower()

    print("Processing:", key)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in" + ext)
        out_path = os.path.join(tmp, "out.mp4")

        # Download
        obj = client.get_object(config.LANDING_BUCKET, key)
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
                config.FORMATTED_BUCKET,
                dst_key,
                data=f,
                length=size,
                content_type="video/mp4",
                metadata = metadata
            )
        print(f"Saved: {config.FORMATTED_BUCKET}/{dst_key}")

def main():
    
    client = get_minio_client()
    for key in list_objects(client, config.LANDING_BUCKET, config.LANDING_VIDEO_PATH):
        try:
            process_video(client, key)
        except S3Error as e:
            print("MinIO error:", e)
        except subprocess.CalledProcessError as e:
            print("ffmpeg error with", key, ":", e)
        except Exception as e:
            print("Unexpected error with", key, ":", e)

if __name__ == "__main__":
    main()
