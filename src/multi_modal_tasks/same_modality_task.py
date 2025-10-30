from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.common.chroma_client import (
    get_client,
    get_text_collection,
    get_image_collection,
    get_video_collection,
    _text_image_ef
)
from src.data_management.exploitation_zone.exploitation_videos import (
    extract_frames_from_file,
    _compute_average_embedding,
)
import src.common.global_variables as config


# Cache Chroma connections
client = get_client()
col_text = get_text_collection(client)  
col_img = get_image_collection(client)
col_video = get_video_collection(client)
BASE_DIR = Path(__file__).resolve().parents[2]  # Project root



def _encode_text_input(text: str):
    """Return the embedding vector for the provided text using the ChromaDB function."""
    clean = text.strip()
    if not clean:
        raise ValueError("Text query must be a non-empty string.")

    embeddings = _text_image_ef([clean])
    if not embeddings:
        raise ValueError("Unable to compute embedding for the provided text.")
    return embeddings[0]


def _encode_image_array(image_array: np.ndarray):
    """Return the embedding vector for the provided RGB image array using the ChromaDB function"""
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Image array must be an RGB image with shape (H, W, 3).")

    rgb_uint8 = np.asarray(image_array, dtype=np.uint8)
    embeddings = _text_image_ef([rgb_uint8])
    if not embeddings:
        raise ValueError("Unable to compute embedding for the provided image.")
    return embeddings[0]


def _encode_image_input(image: Union[str, Path, np.ndarray, Image.Image]):
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


def _load_image(image_path: Path):
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
):
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


def _flatten_query_list(items: Optional[List[Any]]):
    """Return the first nested list (if any) or a flat list to ease formatting."""
    if not items:
        return []
    if isinstance(items, list) and items and isinstance(items[0], list):
        return items[0]
    if isinstance(items, list):
        return items
    return [items]


def _format_text_results_for_display(raw_results: Dict[str, List[Any]]):
    """Build a readable string highlighting each of the returned text documents."""
    documents = _flatten_query_list(raw_results.get("documents"))
    metadatas = _flatten_query_list(raw_results.get("metadatas"))
    distances = _flatten_query_list(raw_results.get("distances"))
    ids = _flatten_query_list(raw_results.get("ids"))

    if not documents:
        return "No similar documents."

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


def _format_image_results_for_display(raw_results: Dict[str, List[Any]]):
    """Format image retrieval output to highlight each match."""
    ids = _flatten_query_list(raw_results.get("ids"))
    metadatas = _flatten_query_list(raw_results.get("metadatas"))
    distances = _flatten_query_list(raw_results.get("distances"))

    if not ids and not metadatas:
        return "No similar images."

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


def _format_video_results_for_display(raw_results: Dict[str, List[Any]]):
    """Format video retrieval output (one embedding per video) for display."""
    ids = _flatten_query_list(raw_results.get("ids"))
    metadatas = _flatten_query_list(raw_results.get("metadatas"))
    distances = _flatten_query_list(raw_results.get("distances"))

    if not ids and not metadatas:
        return "No similar videos."

    sections: List[str] = []
    total = max(len(ids), len(metadatas), len(distances), 1)
    for idx in range(total):
        lines = [f"Video {idx + 1}"]
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


def find_similar_same_modality(
    query: Union[str, Path, Image.Image, np.ndarray],
    *,
    modality: Literal["text", "image", "video"],
    k: int = 5,
    frame_interval_s: float = 1.0,
    max_frames: int = 24,
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
        frame_arrays = [frame for _, _, frame in frames]
        video_embedding = _compute_average_embedding(frame_arrays)
        return col_video.query(
            query_embeddings=[video_embedding],
            n_results=k,
            include=["metadatas", "distances"],
        )

    raise ValueError(f"Unsupported modality '{modality}'. Expected 'text', 'image', or 'video'.")


def find_similar_texts(query: str, *, k_text: int = 5):
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
    )


if __name__ == "__main__":
    selected_modality = "text"  # Options: "text", "image", "video"
    number_of_results = 3

    if selected_modality == "text":
        text_query = "Chicken with garlic and tomatoes"
        results = find_similar_texts(text_query, k_text=number_of_results)
        results = _format_text_results_for_display(results)
    elif selected_modality == "image":
        image = "bread.jpg"
        image_path = (BASE_DIR / f"{config.IMAGE_QUERY_PATH}{image}").resolve()
        image_results = find_similar_images(image_path, k_image=number_of_results)
        results = _format_image_results_for_display(image_results)
    elif selected_modality == "video":
        video = "leche.mp4"
        video_path = (BASE_DIR / f"{config.VIDEO_QUERY_PATH}{video}").resolve()
        video_results = find_similar_videos(
            video_path,
            k_video=number_of_results,
            frame_interval_s=1.0,
            max_frames=50,
        )
        results = _format_video_results_for_display(video_results)
    else:
        raise ValueError("Unsupported modality selected.")

    print(results)
