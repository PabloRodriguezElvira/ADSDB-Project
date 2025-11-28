import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar

from src.multi_modal_tasks.multi_modality_task import find_similar_cross_modality 

BASE_DIR = Path(__file__).resolve().parents[2]

# ------------------- CONFIG -------------------
N_TEXTS = 1000          # how many texts we process
K_CANDIDATES = 100      # how many images are candidates of pairing with a text
LOCAL_OUTPUT = BASE_DIR / "data" / "text_image_matches.json"
MINIO_OUTPUT_KEY = f"{config.FINE_TUNING_PATH}text_image_matches.json"  # upload to MinIO


# ------------------- READ FROM MINIO -------------------

def load_trusted_recipes_texts(n_texts: int) -> List[str]:
    client = get_minio_client()

    key = f"{config.TRUSTED_TEXT_PATH}{config.JSON_NAME}"
    try:
        obj = client.get_object(config.TRUSTED_BUCKET, key)
        chunks: list[bytes] = []
        total_bytes = getattr(obj, "length", None) or getattr(obj, "size", None)
        with ProgressBar(
            total=total_bytes,
            description="Downloading trusted texts",
            unit="B",
            unit_scale=True,
        ) as progress:
            for chunk in obj.stream(1024 * 1024):
                chunks.append(chunk)
                progress.update(len(chunk))

        raw = b"".join(chunks)

        obj.close(); obj.release_conn()
    except S3Error as e:
        raise RuntimeError(f"Error reading {key} from MinIO: {e}")

    data = json.loads(raw.decode("utf-8"))

    if isinstance(data, list):
        texts = data
    elif isinstance(data, dict):
        texts = data.get("root", [])
    else:
        raise ValueError(f"Unexpected JSON format in {key}: {type(data)}")

    if not isinstance(texts, list):
        raise ValueError(f"'root' (or the JSON) must be a list, got {type(texts)}")

    return texts[:n_texts]


# ------------------- MATCH TEXTO - IMAGEN -------------------

def build_trusted_image_key(image_id: str) -> str:
    name = os.path.basename(image_id)

    if not any(name.lower().endswith(ext) for ext in config.IMAGE_EXTENSIONS):
        name = f"{name}.png"

    return f"{config.TRUSTED_IMAGE_PATH}{name}"


def copy_trusted_image_to_finetuning(client, image_id: str, metadata: Dict[str, Any] | None = None) -> str:
    # Prefer the original MinIO key stored during ingestion; fallback to hash-based name
    src_bucket = config.TRUSTED_BUCKET
    src_key = (metadata or {}).get("source_key") or build_trusted_image_key(image_id)

    obj = client.get_object(src_bucket, src_key)
    data = obj.read()
    obj.close(); obj.release_conn()

    dst_basename = os.path.basename(src_key)
    dst_key = f"{config.FINE_TUNING_IMAGE_PATH}{dst_basename}"

    client.put_object(
        config.FINE_TUNING_BUCKET,
        dst_key,
        io.BytesIO(data),
        length=len(data),
        content_type="image/png",
    )

    return dst_key


def match_texts_to_unique_images(
    save_local: bool = True,
    save_to_minio: bool = False,
    copy_images_to_finetuning: bool = True,
) -> List[Dict[str, Any]]:
    """
    - Empareja los primeros N_TEXT textos (trusted) con imágenes únicas (trusted/images).
    - Para cada texto se escoge 1 imagen distinta (sin reutilizar).
    - Solo se copia ESA imagen escogida a fine_tuning/images.
    - Si no se puede encontrar una imagen única para algún texto => lanza error.
    """
    texts = load_trusted_recipes_texts(N_TEXTS)
    total_texts = len(texts)
    print(f"Number of texts to process: {total_texts}")

    client = get_minio_client()
    used_image_ids = set()
    matches: List[Dict[str, Any]] = []

    with ProgressBar(
        total=total_texts,
        description="Emparejando textos con imagenes",
        unit="txt",
        unit_scale=False,
    ) as progress:
        for idx, text in enumerate(texts):
            text_clean = text.strip()
            if not text_clean:
                progress.write(f"[WARN] Texto vacío en índice={idx}, se salta.")
                progress.update(1)
                continue

            # 1) CLIP: texto -> imágenes candidatas
            results = find_similar_cross_modality(
                text_clean,
                relation="text-image",
                k=K_CANDIDATES,
            )

            image_ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            chosen_image_id = None
            chosen_distance = None
            chosen_metadata = None
            finetuning_key = None

            # 2) elegir la primera candidata que NO esté usada
            for img_id, dist, meta in zip(image_ids, distances, metadatas):
                if img_id not in used_image_ids:
                    chosen_image_id = img_id
                    chosen_distance = dist
                    chosen_metadata = meta
                    break

            # si no hay ninguna imagen nueva disponible, no podemos mantener 1:1
            if chosen_image_id is None:
                raise RuntimeError(
                    f"No hay imagen única disponible para el texto índice {idx}. "
                    f"Hay {len(used_image_ids)} imágenes distintas usadas hasta ahora."
                )

            used_image_ids.add(chosen_image_id)

            # 3) copiar SOLO la imagen elegida a fine_tuning/images
            if copy_images_to_finetuning:
                finetuning_key = copy_trusted_image_to_finetuning(client, chosen_image_id, chosen_metadata)

            matches.append(
                {
                    "text_index": idx,
                    "text": text_clean,
                    "image_id": chosen_image_id,
                    "distance": float(chosen_distance) if chosen_distance is not None else None,
                    "image_metadata": chosen_metadata,
                    "fine_tuning_key": finetuning_key,
                }
            )
            progress.update(1)

    # 4) sanity check: 1 imagen por texto
    if len(matches) != len(texts):
        raise RuntimeError(
            f"Matches incompletos: {len(matches)} matches para {len(texts)} textos."
        )

    # 5) guardar JSON
    if save_local:
        LOCAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with LOCAL_OUTPUT.open("w", encoding="utf-8") as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        print(f"JSON de matches guardado en: {LOCAL_OUTPUT}")

    if save_to_minio:
        payload = json.dumps(matches, ensure_ascii=False, indent=2).encode("utf-8")
        client.put_object(
            config.FINE_TUNING_BUCKET,
            MINIO_OUTPUT_KEY,
            io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )
        print(f"JSON de matches subido a MinIO en: {MINIO_OUTPUT_KEY}")

    return matches


if __name__ == "__main__":
    matches = match_texts_to_unique_images(
        save_local=True,
        save_to_minio=True, 
    )
    print(f"Total de matches: {len(matches)}")
