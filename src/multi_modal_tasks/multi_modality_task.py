from __future__ import annotations
from pathlib import Path
from typing import Literal
import numpy as np
from PIL import Image

from src.common.chroma_client import (
    get_client,
    get_text_collection,
    get_image_collection,
    get_video_collection,
)
from src.data_management.exploitation_zone.exploitation_videos import (
    _compute_average_embedding,
)
# import functions used in the same modality task 
from src.multi_modal_tasks.same_modality_task import (
    _encode_text_input,
    _encode_image_input,
    _extract_frames,
    _format_text_results_for_display,
    _format_image_results_for_display,
    _format_video_results_for_display,
)
import src.common.global_variables as config


# Cache Chroma connections
client = get_client()
col_text = get_text_collection(client)
col_img = get_image_collection(client)
col_video = get_video_collection(client)
BASE_DIR = Path(__file__).resolve().parents[2]  # Project root



def find_similar_cross_modality(
    query: Union[str, Path, Image.Image, np.ndarray],
    *,
    relation: Literal[
        "text-image", "text-video",
        "image-text", "image-video",
        "video-text", "video-image"
    ],
    k: int = 5,
    frame_interval_s: float = 1.0,
    max_frames: int = 24,
):
    """Perform cross-modality search using CLIP embeddings"""
    
    relation = relation.lower()

    # Compute embedding for the source modality 
    if relation.startswith("text-"):
        embedding = _encode_text_input(query)

    elif relation.startswith("image-"):
        embedding, _ = _encode_image_input(query)

    elif relation.startswith("video-"):
        video_path = Path(query)
        frames = _extract_frames(video_path, frame_interval_s=frame_interval_s, max_frames=max_frames)
        frame_arrays = [frame for _, _, frame in frames]
        embedding = _compute_average_embedding(frame_arrays)

    else:
        raise ValueError(f"Unsupported source modality in relation '{relation}'.")


    # Select the target collection and the metadatas
    if relation.endswith("-text"):
        collection = col_text
        include_fields = ["documents", "metadatas", "distances"]

    elif relation.endswith("-image"):
        collection = col_img
        include_fields = ["metadatas", "distances"]

    elif relation.endswith("-video"):
        collection = col_video
        include_fields = ["metadatas", "distances"]

    else:
        raise ValueError(f"Unsupported target modality in relation '{relation}'.")


    # Query the target Chroma collection to retrieve the k most similar items to the input embedding.
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=include_fields,
    )

    return results


if __name__ == "__main__":
    "Main entry point: select which cross-modality search to run"
    relation = "image-text" # Options: "text-image", "text-video", "image-text", "image-video", "video-text", "video-image"
    number_of_results = 3

    if relation == "text-image":
        text_query = "Shredded pork cooked slowly in Texas style."
        results = find_similar_cross_modality(text_query, relation=relation, k=number_of_results)
        print(_format_image_results_for_display(results))

    elif relation == "text-video":
        text_query = "Slow Cooker Texas Pulled Pork"
        results = find_similar_cross_modality(text_query, relation=relation, k=number_of_results)
        print(_format_video_results_for_display(results))

    elif relation == "image-text":
        image = "pulled_pork.png"
        image_path = (BASE_DIR / f"{config.IMAGE_QUERY_PATH}{image}").resolve()
        results = find_similar_cross_modality(image_path, relation=relation, k=number_of_results)
        print(_format_text_results_for_display(results))

    elif relation == "image-video":
        image = "bread.jpg"
        image_path = (BASE_DIR / f"{config.IMAGE_QUERY_PATH}{image}").resolve()
        results = find_similar_cross_modality(image_path, relation=relation, k=number_of_results)
        print(_format_video_results_for_display(results))

    elif relation == "video-text":
        video = "leche.mp4"
        video_path = (BASE_DIR / f"{config.VIDEO_QUERY_PATH}{video}").resolve()
        results = find_similar_cross_modality(video_path, relation=relation, k=number_of_results)
        print(_format_text_results_for_display(results))

    elif relation == "video-image":
        video = "leche.mp4"
        video_path = (BASE_DIR / f"{config.VIDEO_QUERY_PATH}{video}").resolve()
        results = find_similar_cross_modality(video_path, relation=relation, k=number_of_results)
        print(_format_image_results_for_display(results))

    else:
        raise ValueError(f"Unsupported relation: {relation}")
