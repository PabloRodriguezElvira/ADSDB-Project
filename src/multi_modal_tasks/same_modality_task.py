from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.common.chroma_client import (
    get_client,
    get_image_collection,
    get_text_collection,
    _text_ef,
    _image_ef,
)
from src.data_management.exploitation_zone.exploitation_videos import (
    extract_frames_from_file,
)
import src.common.global_variables as config


# Cache Chroma connections just like in generative_task.py
client = get_client()
col_text = get_text_collection(client)
col_img = get_image_collection(client)
BASE_DIR = Path(__file__).resolve().parents[2]  # Project root



def _encode_text_input(text: str) -> List[float]:
    """Return the embedding vector for the provided text using the ChromaDB function."""
    clean = text.strip()
    if not clean:
        raise ValueError("Text query must be a non-empty string.")

    embeddings = _text_ef([clean])
    if not embeddings:
        raise ValueError("Unable to compute embedding for the provided text.")
    return embeddings[0]


def _encode_image_array(image_array: np.ndarray) -> List[float]:
    """Return the embedding vector for the provided RGB image array using the ChromaDB function"""
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Image array must be an RGB image with shape (H, W, 3).")

    rgb_uint8 = np.asarray(image_array, dtype=np.uint8)
    embeddings = _image_ef([rgb_uint8])
    if not embeddings:
        raise ValueError("Unable to compute embedding for the provided image.")
    return embeddings[0]


def _encode_image_input(image: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[List[float], np.ndarray]:
    """Load and encode an image, returning the embedding and the RGB array."""
    if isinstance(image, (str, Path)):
        array = _load_image(Path(image))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError("Unsupported image input type.")

    embedding = _encode_image_array(array)
    return embedding, array


def _load_image(image_path: Path) -> np.ndarray:
    """Load an image from disk into an RGB NumPy array."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported image format: {image_path}") from exc


def _extract_frames(
    video_path: Path,
    *,
    frame_interval_s: float = 1.0,
    max_frames: int = 24,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Sample frames from a video every `frame_interval_s` seconds.

    Returns a list of RGB NumPy arrays ready for Chroma queries.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be greater than zero.")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")

    frames: List[Tuple[int, float, np.ndarray]] = []
    for frame_idx, timestamp_s, frame_bgr in extract_frames_from_file(
        str(video_path),
        frame_interval_s=frame_interval_s,
        max_frames=max_frames,
    ):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append((frame_idx, float(timestamp_s), frame_rgb))

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def _aggregate_video_matches(
    frames: List[Tuple[int, float, np.ndarray]],
    raw: Dict[str, List[Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:

    """Aggregate frame-level matches into video-level results."""
    matches: Dict[str, Dict[str, Any]] = {}
    ids_lists = raw.get("ids", [])
    distances_lists = raw.get("distances", [])
    metadata_lists = raw.get("metadatas", [])

    for frame_idx, frame_matches in enumerate(ids_lists):
        if frame_idx >= len(frames):
            break
        query_frame_idx, query_timestamp_s, _ = frames[frame_idx]
        metadata_list = metadata_lists[frame_idx] if frame_idx < len(metadata_lists) else []
        distances_list = distances_lists[frame_idx] if frame_idx < len(distances_lists) else []

        for match_pos, match_id in enumerate(frame_matches):
            metadata = metadata_list[match_pos] if match_pos < len(metadata_list) else {}
            distance_raw = distances_list[match_pos] if match_pos < len(distances_list) else None
            try:
                distance = float(distance_raw) if distance_raw is not None else float("inf")
            except (TypeError, ValueError):
                distance = float("inf")

            video_key = metadata.get("video_key") or metadata.get("source_key") or match_id
            entry = matches.setdefault(
                video_key,
                {
                    "video_key": video_key,
                    "best_distance": float("inf"),
                    "matches": [],
                },
            )
            entry["best_distance"] = min(entry["best_distance"], distance)
            entry["matches"].append(
                {
                    "id": match_id,
                    "distance": distance,
                    "metadata": metadata,
                    "query_frame_idx": query_frame_idx,
                    "query_timestamp_s": query_timestamp_s,
                }
            )

    sorted_results = sorted(matches.values(), key=lambda item: item["best_distance"])
    return sorted_results[:limit]


def _flatten_query_list(items: Optional[List[Any]]) -> List[Any]:
    """Return the first nested list (if any) or a flat list to ease formatting."""
    if not items:
        return []
    if isinstance(items, list) and items and isinstance(items[0], list):
        return items[0]
    if isinstance(items, list):
        return items
    return [items]


def _format_text_results_for_display(raw_results: Dict[str, List[Any]]) -> str:
    """Build a readable string highlighting each of the returned text documents."""
    documents = _flatten_query_list(raw_results.get("documents"))
    metadatas = _flatten_query_list(raw_results.get("metadatas"))
    distances = _flatten_query_list(raw_results.get("distances"))
    ids = _flatten_query_list(raw_results.get("ids"))

    if not documents:
        return "No se encontraron documentos similares."

    sections: List[str] = []
    total = max(len(documents), len(metadatas), len(distances), len(ids))
    for idx in range(total):
        lines = [f"Resultado {idx + 1}"]
        if idx < len(ids) and ids[idx] is not None:
            lines.append(f"ID: {ids[idx]}")
        if idx < len(distances):
            try:
                distance_value = float(distances[idx])
                lines.append(f"Distancia: {distance_value:.4f}")
            except (TypeError, ValueError):
                lines.append(f"Distancia: {distances[idx]}")
        if idx < len(metadatas) and metadatas[idx] is not None:
            lines.append(f"Metadatos: {metadatas[idx]}")
        if idx < len(documents):
            lines.append("Documento:")
            lines.append(str(documents[idx]))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _format_image_results_for_display(raw_results: Dict[str, List[Any]]) -> str:
    """Format image retrieval output to highlight each match."""
    ids = _flatten_query_list(raw_results.get("ids"))
    metadatas = _flatten_query_list(raw_results.get("metadatas"))
    distances = _flatten_query_list(raw_results.get("distances"))

    if not ids and not metadatas:
        return "No se encontraron imágenes similares."

    total = max(len(ids), len(metadatas), len(distances), 1)
    sections: List[str] = []
    for idx in range(total):
        lines = [f"Resultado {idx + 1}"]
        if idx < len(ids) and ids[idx] is not None:
            lines.append(f"ID: {ids[idx]}")
        if idx < len(distances):
            try:
                distance_value = float(distances[idx])
                lines.append(f"Distancia: {distance_value:.4f}")
            except (TypeError, ValueError):
                lines.append(f"Distancia: {distances[idx]}")
        if idx < len(metadatas) and metadatas[idx] is not None:
            lines.append(f"Metadatos: {metadatas[idx]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _format_video_results_for_display(video_results: List[Dict[str, Any]]) -> str:
    """Summarize video matches including their best score and a sample frame."""
    if not video_results:
        return "No se encontraron vídeos similares."

    sections: List[str] = []
    for idx, entry in enumerate(video_results, start=1):
        lines = [
            f"Video {idx}",
            f"Video key: {entry.get('video_key')}",
            f"Mejor distancia: {entry.get('best_distance')}",
            f"Total coincidencias de frames: {len(entry.get('matches', []))}",
        ]
        matches = entry.get("matches", [])
        if matches:
            best = matches[0]
            lines.append(
                "Coincidencia destacada: "
                f"id={best.get('id')}, distancia={best.get('distance')}, "
                f"frame={best.get('query_frame_idx')} (t={best.get('query_timestamp_s')}s)"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def find_similar_same_modality(
    query: Union[str, Path, Image.Image, np.ndarray],
    *,
    modality: Literal["text", "image", "video"],
    k: int = 5,
    frame_interval_s: float = 1.0,
    max_frames: int = 24,
    k_per_frame: Optional[int] = None,
):
    """
    Encode a text, image, or video using the configured Chroma embedding functions and
    return the `k` closest elements in the corresponding collection.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if modality == "text":
        if not isinstance(query, str):
            raise TypeError("Text modality expects a string query.")
        embedding = _encode_text_input(query)
        return col_text.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    if modality == "image":
        embedding, _ = _encode_image_input(query)
        return col_img.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["metadatas", "distances"],
        )

    if modality == "video":
        video_path = Path(query) if not isinstance(query, Path) else query
        frames = _extract_frames(
            video_path,
            frame_interval_s=frame_interval_s,
            max_frames=max_frames,
        )
        if not frames:
            return []

        if k_per_frame is None:
            k_per_frame = max(k, 5)
        if k_per_frame <= 0:
            raise ValueError("k_per_frame must be greater than zero.")

        frame_arrays = [frame for _, _, frame in frames]
        print("LEN: ", col_img.count())
        raw = col_img.query(
            query_images=frame_arrays,
            n_results=k_per_frame,
            include=["metadatas", "distances"],
        )
        return _aggregate_video_matches(frames, raw, limit=k)

    raise ValueError(f"Unsupported modality '{modality}'. Expected 'text', 'image', or 'video'.")


def find_similar_texts(query: str, *, k_text: int = 5) -> Dict[str, List[Any]]:
    """Wraps the function `find_similar_same_modality` for text queries."""
    return find_similar_same_modality(query, modality="text", k=k_text)


def find_similar_images(
    image: Union[str, Path, Image.Image, np.ndarray],
    *,
    k_image: int = 5,
):
    """Wraps the function 'find_similar_same_modality` for image queries."""
    return find_similar_same_modality(image, modality="image", k=k_image)


def find_similar_videos(
    video: Union[str, Path],
    *,
    k_video: int = 5,
    k_per_frame: Optional[int] = None,
    frame_interval_s: float = 1.0,
    max_frames: int = 24,
):
    """Wraps the function `find_similar_same_modality` for video queries."""
    return find_similar_same_modality(
        video,
        modality="video",
        k=k_video,
        frame_interval_s=frame_interval_s,
        max_frames=max_frames,
        k_per_frame=k_per_frame,
    )


if __name__ == "__main__":
    selected_modality = "video"  # Options: "text", "image", "video"

    if selected_modality == "text":
        text_query = "Slow Cooker Texas Pulled Pork"
        results = find_similar_texts(text_query, k_text=3)
        results = _format_text_results_for_display(results)
    elif selected_modality == "image":
        image = "bread.jpg"
        image_path = (BASE_DIR / f"{config.IMAGE_QUERY_PATH}{image}").resolve()
        image_results = find_similar_images(image_path, k_image=10)
        results = _format_image_results_for_display(image_results)
    elif selected_modality == "video":
        video = "tomates.mp4"
        video_path = (BASE_DIR / f"{config.VIDEO_QUERY_PATH}{video}").resolve()
        video_results = find_similar_videos(
            video_path,
            k_video=1,
            frame_interval_s=1.0,
            max_frames=50,
            k_per_frame=None,
        )
        results = _format_video_results_for_display(video_results)
    else:
        raise ValueError("Unsupported modality selected.")

    print(results)
