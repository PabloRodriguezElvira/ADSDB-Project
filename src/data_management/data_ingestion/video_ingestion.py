import os
import requests
from pathlib import Path
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from datetime import datetime
from zoneinfo import ZoneInfo
from src.common.minio_client import get_minio_client

# Config
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"


# Pick the most suitable file variant for a Pexels video, favoring HD (<= 1080p) and otherwise the option with the greatest width.
def pick_best_video_file(video_files: list) -> dict | None:
    if not video_files:
        return None

    hd_candidates = [vf for vf in video_files if vf.get("quality") == "hd" and vf.get("width", 0) <= 1920]
    if hd_candidates:
        return max(hd_candidates, key=lambda v: v.get("bitrate", 0) or v.get("width", 0))

    return max(video_files, key=lambda v: (v.get("width", 0), v.get("bitrate", 0) or 0))


# Collect the requested amount of videos from Pexels, paginating the API (80 items max per page).
def search_pexels_videos(query: str, videos_amount: int, per_page: int = 80):
    assert PEXELS_API_KEY, "PEXELS_API_KEY not found in environment."
    headers = {"Authorization": PEXELS_API_KEY}
    results = []
    page = 1
    remaining = videos_amount

    while remaining > 0:
        batch_size = min(per_page, remaining)
        params = {"query": query, "per_page": batch_size, "page": page, "orientation": "landscape"}
        resp = requests.get(PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        videos = data.get("videos", [])
        if not videos:
            break
        results.extend(videos)
        remaining = videos_amount - len(results)
        page += 1

    return results[:videos_amount]


# Stream a remote video into MinIO without persisting it locally; when the size is unknown, we use BytesIO to buffer it in memory.
def upload_video_streaming(client: Minio, bucket: str, folder: str, filename: str, url: str) -> bool:
    object_name = f"{folder}/{filename}"

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        content_length = response.headers.get("Content-Length")

        metadata = {
            "x-amz-meta-data-source": "video",
            "x-amz-meta-ingested-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        }

        # If we know the length, we can upload response.raw using put_object.
        if content_length is not None:
            client.put_object(
                bucket,
                object_name,
                response.raw,
                length=int(content_length),
                metadata=metadata,
            )
            print(f"[OK] Uploaded {url} to s3://{bucket}/{object_name}")
            return True

        # If we don't know the length we have to use the buffer to store the image and know it.
        buffer = BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                buffer.write(chunk)

        size = buffer.getbuffer().nbytes
        if size == 0:
            print(f"[WARN] Empty download for {url}, skipping.")
            return False

        buffer.seek(0)
        client.put_object(
            bucket,
            object_name,
            buffer,
            length=size,
            metadata=metadata,
        )
        print(f"[OK] Uploaded {url} to s3://{bucket}/{object_name}")
        return True



# Coordinate the ingestion: search on Pexels, choose the best rendition, and push each video directly to MinIO.
def main():
    query = "healthy food cooking"
    videos_amount = 3
    bucket = "landing-zone"
    folder = "temporal_landing"

    client = get_minio_client()

    print(f"[INFO] Searching Pexels videos: '{query}' (max {videos_amount})")
    videos = search_pexels_videos(query=query, videos_amount=videos_amount)

    # Iterate through the videos, choose the best video file and upload it.
    uploaded_count = 0
    for video in videos:
        vid_id = video.get("id")
        best = pick_best_video_file(video.get("video_files", []))
        if not best:
            print(f"[WARN] Video {vid_id} has no files available, skipping.")
            continue

        url = best.get("link")
        file_type = (best.get("file_type") or "").lower()

        # Try to get the extension from the file_type or the url. If not possible, we use .mp4.
        if file_type.startswith("video/"):
            ext = f".{file_type.split('/', 1)[1]}"
        else:
            ext = Path(url).suffix or ".mp4"

        width = best.get("width")
        height = best.get("height")
        quality = best.get("quality") or "unknown"
        filename = f"pexels_{vid_id}_{quality}_{width}x{height}{ext}"

        try:
            print(f"[INFO] Uploading {url} as {filename}")
            if upload_video_streaming(client, bucket, folder, filename, url):
                uploaded_count += 1
        except Exception as exc:
            print(f"[ERROR] Direct upload failed for {url}: {exc}")

    if uploaded_count == 0:
        print("[WARN] No videos were uploaded to the temporal bucket.")
    else:
        print(f"[OK] Uploaded {uploaded_count} videos to the temporal bucket.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[CONFIG] {exc}")
    except S3Error as exc:
        print(f"MinIO error: {exc}")
    except requests.HTTPError as exc:
        print(f"HTTP error from Pexels: {exc}")
