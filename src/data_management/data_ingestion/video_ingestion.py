import requests
from pathlib import Path
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

from src.common.minio_client import get_minio_client
from src.common.progress_bar import ProgressBar
import src.common.global_variables as config



def pick_best_video_file(video_files: list):
    """
    Select the most suitable video file from a Pexels video entry.
    Prefer HD (<=1080p) files; otherwise, choose the file with the largest width/bitrate.
    """

    if not video_files:
        return None

    hd_candidates = [vf for vf in video_files if vf.get("quality") == "hd" and vf.get("width", 0) <= 1920]

    if hd_candidates:
        return max(hd_candidates, key=lambda v: v.get("bitrate", 0) or v.get("width", 0))

    return max(video_files, key=lambda v: (v.get("width", 0), v.get("bitrate", 0) or 0))


def search_pexels_videos(query: str, videos_amount: int, per_page: int = 80):
    """
    Query the Pexels API to collect a given number of videos matching the search query.
    Handles pagination automatically (max 80 items per page).
    """

    assert config.PEXELS_API_KEY, "PEXELS_API_KEY not found in environment."
    headers = {"Authorization": config.PEXELS_API_KEY}
    results = []
    page = 1
    remaining = videos_amount

    while remaining > 0:
        batch_size = min(per_page, remaining)
        params = {"query": query, "per_page": batch_size, "page": page, "orientation": "landscape"}
        resp = requests.get(config.PEXELS_VIDEO_SEARCH_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        videos = data.get("videos", [])
        if not videos:
            break
        results.extend(videos)
        remaining = videos_amount - len(results)
        page += 1

    return results[:videos_amount]


def upload_video_streaming(
    client: Minio,
    bucket: str,
    folder: str,
    filename: str,
    url: str,
    progress: Optional[ProgressBar] = None,
):
    """
    Stream a remote video from Pexels directly into MinIO.
    If content length is known, stream directly; otherwise, buffer with BytesIO before uploading.
    """

    object_name = f"{folder}{filename}"

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        content_length = response.headers.get("Content-Length")

        metadata = {
            "x-amz-meta-data-source": "video",
            "x-amz-meta-ingested-at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(),
        }

        # If the size is known, upload the stream directly.
        if content_length is not None:
            client.put_object(
                bucket,
                object_name,
                response.raw,
                length=int(content_length),
                metadata=metadata,
            )
            return True

        # Otherwise, buffer the content in memory to determine its size.
        buffer = BytesIO()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                buffer.write(chunk)

        size = buffer.getbuffer().nbytes
        if size == 0:
            return False

        buffer.seek(0)
        client.put_object(
            bucket,
            object_name,
            buffer,
            length=size,
            metadata=metadata,
        )

        return True


def main():
    """
    Search for videos on Pexels, select the best file for each result,
    and upload them directly to the MinIO temporal landing bucket.
    """
    
    query = "healthy food cooking"
    videos_amount = 50

    client = get_minio_client()

    print(f"[INFO] Searching Pexels videos: '{query}' (max {videos_amount})")
    videos = search_pexels_videos(query=query, videos_amount=videos_amount)

    uploaded_count = 0
    upload_queue: list[tuple[str, str]] = []

    for video in videos:
        vid_id = video.get("id")
        best = pick_best_video_file(video.get("video_files", []))
        if not best:
            print(f"[WARN] Video {vid_id} has no files available, skipping.")
            continue

        url = best.get("link")
        file_type = (best.get("file_type") or "").lower()

        # Try to infer extension from MIME type. If not, set default to .mp4
        if file_type.startswith("video/"):
            ext = f".{file_type.split('/', 1)[1]}"
        else:
            ext = Path(url).suffix or ".mp4"

        width = best.get("width")
        height = best.get("height")
        quality = best.get("quality") or "unknown"
        filename = f"pexels_{vid_id}_{quality}_{width}x{height}{ext}"

        upload_queue.append((filename, url))

    if not upload_queue:
        print("[WARN] No videos were queued for upload to the temporal bucket.")
        return

    with ProgressBar(
        total=len(upload_queue),
        description="Uploading video dataset",
        unit="video",
        unit_scale=False,
        unit_divisor=1,
    ) as progress:
        for filename, url in upload_queue:
            progress.set_description(f"Uploading {filename}", refresh=True)
            try:
                if upload_video_streaming(
                    client,
                    config.LANDING_BUCKET,
                    config.LANDING_TEMPORAL_PATH,
                    filename,
                    url,
                    progress,
                ):
                    uploaded_count += 1
            except Exception as exc:
                progress.write(f"[ERROR] Direct upload failed for {url}: {exc}")
            finally:
                progress.update(1)

    if uploaded_count == 0:
        print("[WARN] No videos were uploaded to the temporal bucket.")


if __name__ == "__main__":
    """
    Script entry point: handles configuration, MinIO, and HTTP errors.
    """

    try:
        main()
    except AssertionError as exc:
        print(f"[CONFIG] {exc}")
    except S3Error as exc:
        print(f"MinIO error: {exc}")
    except requests.HTTPError as exc:
        print(f"HTTP error from Pexels: {exc}")
