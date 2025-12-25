import io
import json
import random
from typing import List, Tuple

from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config

# Fuente: JSON completo generado en augmentation_zone.py
INPUT_BUCKET = config.AUGMENTATION_BUCKET
INPUT_KEY = f"{config.AUGMENTATION_PATH}augmented_text_image_matches.json"

# Destino: índices de cada split en texto plano
OUTPUT_BUCKET = config.SPLIT_BUCKET
TRAIN_KEY = f"{config.SPLIT_PATH}train.txt"
TEST_KEY = f"{config.SPLIT_PATH}test.txt"
DEV_KEY = f"{config.SPLIT_PATH}dev.txt"


def load_augmented_samples() -> List[dict]:
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


def split_indices(
    num_samples: int,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Devuelve listas de índices para train/test/dev con seed fija."""
    if num_samples <= 0:
        return [], [], []

    all_indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    n_train = int(num_samples * train_ratio)
    n_test = int(num_samples * test_ratio)

    train_idx = all_indices[:n_train]
    test_idx = all_indices[n_train:n_train + n_test]
    dev_idx = all_indices[n_train + n_test:]

    return train_idx, test_idx, dev_idx


def _save_indices_to_minio(key: str, indices: List[int]) -> None:
    client = get_minio_client()
    payload = "\n".join(str(i) for i in indices).encode("utf-8")
    try:
        client.put_object(
            OUTPUT_BUCKET,
            key,
            io.BytesIO(payload),
            length=len(payload),
            content_type="text/plain",
        )
    except S3Error as e:
        raise RuntimeError(f"Error subiendo {key} a MinIO: {e}")


def split_augmented_json(
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    samples = load_augmented_samples()
    train_idx, test_idx, dev_idx = split_indices(
        len(samples),
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    _save_indices_to_minio(TRAIN_KEY, train_idx)
    _save_indices_to_minio(TEST_KEY, test_idx)
    _save_indices_to_minio(DEV_KEY, dev_idx)

    return train_idx, test_idx, dev_idx


if __name__ == "__main__":
    train_idx, test_idx, dev_idx = split_augmented_json()
    print(
        f"Splits generados -> TRAIN: {len(train_idx)}, TEST: {len(test_idx)}, DEV: {len(dev_idx)}"
    )
