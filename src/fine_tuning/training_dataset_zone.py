import io
import json
import base64
from typing import List, Dict, Any

from PIL import Image
from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar

# ------------------- CONFIG -------------------

# Fuente: JSON de augmentation
INPUT_BUCKET = config.AUGMENTATION_BUCKET
INPUT_KEY = f"{config.AUGMENTATION_PATH}augmented_text_image_matches.json"

# Fuente: splits (indices)
SPLIT_BUCKET = config.SPLIT_BUCKET
TRAIN_KEY = f"{config.SPLIT_PATH}train.txt"
TEST_KEY = f"{config.SPLIT_PATH}test.txt"
DEV_KEY = f"{config.SPLIT_PATH}dev.txt"

# Destino: training dataset
OUTPUT_BUCKET = config.TRAINING_DATASET_BUCKET
TRAIN_PREFIX = config.TRAINING_TRAIN
TEST_PREFIX = config.TRAINING_TEST
DEV_PREFIX = config.TRAINING_DEV


def load_augmented_samples() -> List[Dict[str, Any]]:
    client = get_minio_client()
    try:
        obj = client.get_object(INPUT_BUCKET, INPUT_KEY)
        raw = obj.read()
        obj.close(); obj.release_conn()
    except S3Error as e:
        raise RuntimeError(
            f"Error leyendo JSON de augmentation {INPUT_BUCKET}/{INPUT_KEY}: {e}"
        )

    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"El JSON de augmentation debe ser una lista, se obtuvo {type(data)}"
        )
    return data


def load_split_indices(key: str) -> List[int]:
    client = get_minio_client()
    try:
        obj = client.get_object(SPLIT_BUCKET, key)
        raw = obj.read()
        obj.close(); obj.release_conn()
    except S3Error as e:
        raise RuntimeError(f"Error leyendo split {SPLIT_BUCKET}/{key}: {e}")

    lines = [line.strip() for line in raw.decode("utf-8").splitlines()]
    indices: List[int] = []
    for line in lines:
        if not line:
            continue
        try:
            indices.append(int(line))
        except ValueError as e:
            raise ValueError(f"Indice invalido en {key}: {line}") from e
    return indices


def _decode_base64_to_png(image_base64: str) -> bytes:
    try:
        raw = base64.b64decode(image_base64)
    except Exception as e:
        raise ValueError(f"Base64 invalido: {e}") from e

    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        raise ValueError(f"No se pudo decodificar la imagen: {e}") from e


def _export_split(
    split_name: str,
    split_prefix: str,
    indices: List[int],
    samples: List[Dict[str, Any]],
    client,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    images_prefix = f"{split_prefix}images/"

    with ProgressBar(
        total=len(indices),
        description=f"Volcando split {split_name}",
        unit="img",
        unit_scale=False,
    ) as progress:
        for idx in indices:
            if idx < 0 or idx >= len(samples):
                progress.write(f"[WARN] indice fuera de rango: {idx}")
                progress.update(1)
                continue

            sample = samples[idx]
            text = sample.get("text", "")
            image_base64 = sample.get("image_base64")

            if not image_base64:
                progress.write(f"[WARN] No hay image_base64 para id={idx}")
                progress.update(1)
                continue

            try:
                image_bytes = _decode_base64_to_png(image_base64)
            except ValueError as e:
                progress.write(f"[WARN] Fallo en id={idx}: {e}")
                progress.update(1)
                continue

            image_key = f"{images_prefix}{idx}.png"

            try:
                client.put_object(
                    OUTPUT_BUCKET,
                    image_key,
                    io.BytesIO(image_bytes),
                    length=len(image_bytes),
                    content_type="image/png",
                )
            except S3Error as e:
                raise RuntimeError(
                    f"Error subiendo imagen {OUTPUT_BUCKET}/{image_key}: {e}"
                )

            matches.append(
                {
                    "id": idx,
                    "text": text,
                    "image_path": image_key,
                }
            )
            progress.update(1)

    matches_key = f"{split_prefix}matches.json"
    payload = json.dumps(matches, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        client.put_object(
            OUTPUT_BUCKET,
            matches_key,
            io.BytesIO(payload),
            length=len(payload),
            content_type="application/json",
        )
    except S3Error as e:
        raise RuntimeError(
            f"Error subiendo matches {OUTPUT_BUCKET}/{matches_key}: {e}"
        )

    return matches


def build_training_dataset() -> Dict[str, int]:
    samples = load_augmented_samples()
    train_idx = load_split_indices(TRAIN_KEY)
    test_idx = load_split_indices(TEST_KEY)
    dev_idx = load_split_indices(DEV_KEY)

    client = get_minio_client()

    train_matches = _export_split("train", TRAIN_PREFIX, train_idx, samples, client)
    test_matches = _export_split("test", TEST_PREFIX, test_idx, samples, client)
    dev_matches = _export_split("dev", DEV_PREFIX, dev_idx, samples, client)

    return {
        "train": len(train_matches),
        "test": len(test_matches),
        "dev": len(dev_matches),
    }


if __name__ == "__main__":
    counts = build_training_dataset()
    print(
        "Training dataset generado -> "
        f"TRAIN: {counts['train']}, TEST: {counts['test']}, DEV: {counts['dev']}"
    )
