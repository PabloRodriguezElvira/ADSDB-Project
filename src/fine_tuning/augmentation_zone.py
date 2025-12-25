import io
import json
import base64
import random
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image, ImageEnhance, ImageOps
from minio.error import S3Error

from src.common.minio_client import get_minio_client
import src.common.global_variables as config
from src.common.progress_bar import ProgressBar

BASE_DIR = Path(__file__).resolve().parents[2]

# ------------------- CONFIG -------------------

# JSON de matches original (generado por tu script actual)
INPUT_BUCKET = config.FINE_TUNING_BUCKET
INPUT_MATCHES_KEY = f"{config.FINE_TUNING_PATH}text_image_matches.json"

# Salida (un único JSON) en MinIO y local
AUGMENTED_JSON_KEY = f"{config.AUGMENTATION_PATH}augmented_text_image_matches.json"
LOCAL_AUGMENTED_JSON = BASE_DIR / "data" / "augmented_text_image_matches.json"

# Número de augmentations por par texto-imagen
N_AUG_PER_SAMPLE = 1


# ------------------- UTILIDADES BASE64/IMAGEN -------------------

def _base64_to_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str.encode("utf-8"))
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _random_image_augmentation(img: Image.Image) -> tuple[Image.Image, dict]:
    """
    Augmentation para CLIP:
      - random crop ligero
      - rotación aleatoria [-15, 15]
      - flip horizontal opcional
      - cambios pequeños de brillo, contraste y color
    """
    aug = img.copy()
    w, h = aug.size

    # ----- Random crop ligero (90–100% del tamaño original) -----
    scale = random.uniform(0.9, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    left   = random.randint(0, w - new_w)
    top    = random.randint(0, h - new_h)
    right  = left + new_w
    bottom = top + new_h
    aug = aug.crop((left, top, right, bottom))
    # Opcional: reescalar de nuevo al tamaño original
    aug = aug.resize((w, h), Image.BICUBIC)

    # ----- Rotación -----
    angle = random.uniform(-15, 15)
    aug = aug.rotate(angle, resample=Image.BICUBIC, expand=False)

    # ----- Flip horizontal -----
    hflip = False
    if random.random() < 0.5:
        aug = ImageOps.mirror(aug)
        hflip = True

    # ----- Color jitter -----
    brightness_factor = random.uniform(0.8, 1.2)
    contrast_factor   = random.uniform(0.8, 1.2)
    color_factor      = random.uniform(0.8, 1.2)

    aug = ImageEnhance.Brightness(aug).enhance(brightness_factor)
    aug = ImageEnhance.Contrast(aug).enhance(contrast_factor)
    aug = ImageEnhance.Color(aug).enhance(color_factor)

    meta = {
        "crop_scale": scale,
        "crop_box": (left, top, right, bottom),
        "rotation_deg": angle,
        "horizontal_flip": hflip,
        "brightness_factor": brightness_factor,
        "contrast_factor": contrast_factor,
        "color_factor": color_factor,
    }
    return aug, meta


# ------------------- (OPCIONAL) AUGMENTACIÓN DE TEXTO -------------------

def augment_text(text: str, aug_idx: int) -> str:
    """
    Hook para augmentación de texto.
    Ahora mismo devuelve el mismo texto.
    """
    return text


# ------------------- LECTURA DEL JSON ORIGINAL -------------------

def load_matches_from_minio() -> List[Dict[str, Any]]:
    client = get_minio_client()

    try:
        obj = client.get_object(INPUT_BUCKET, INPUT_MATCHES_KEY)
        raw = obj.read()
        obj.close(); obj.release_conn()
    except S3Error as e:
        raise RuntimeError(
            f"Error leyendo JSON de matches {INPUT_BUCKET}/{INPUT_MATCHES_KEY}: {e}"
        )

    matches = json.loads(raw.decode("utf-8"))
    if not isinstance(matches, list):
        raise ValueError(
            f"El JSON de matches debe ser una lista, se obtuvo {type(matches)}"
        )
    return matches


# ------------------- PIPELINE PRINCIPAL -------------------
def augment_dataset(
    n_aug_per_sample: int = N_AUG_PER_SAMPLE,
    save_local: bool = False,
    save_to_minio: bool = True,
) -> List[Dict[str, Any]]:
    """
    - Lee `text_image_matches.json` (con `image_base64`) desde MinIO.
    - Construye un dataset con:
        * todas las muestras ORIGINALES
        * y n_aug_per_sample augmentations por cada una
      => tamaño total = num_original * (1 + n_aug_per_sample)
    - Genera un único JSON con todas las muestras (sin split).
    """
    original_matches = load_matches_from_minio()
    num_original = len(original_matches)

    all_samples: List[Dict[str, Any]] = []

    # 1) Construir dataset (original + augmentations)
    with ProgressBar(
        total=num_original,
        description="Generando JSON con originales + augmentations",
        unit="sample",
        unit_scale=False,
    ) as progress:
        for i, match in enumerate(original_matches):
            text = match.get("text", "")
            text_index = match.get("text_index", i)  # fallback a i si no está
            image_base64 = match.get("image_base64")

            if not image_base64:
                progress.write(
                    f"[WARN] No hay 'image_base64' para text_index={text_index}, se salta."
                )
                progress.update(1)
                continue

            # --------- Entrada ORIGINAL ---------
            original_entry = {
                "original_index": i,
                "original_text_index": text_index,
                "original_image_id": match.get("image_id"),

                "text": text,
                "image_base64": image_base64,

                "image_metadata": match.get("image_metadata"),
                "distance": match.get("distance"),

                "is_augmented": False,
                "augmentation_index": None,
                "augmentation_metadata": None,
            }
            all_samples.append(original_entry)

            # --------- Augmentations ---------
            try:
                img = _base64_to_image(image_base64)
            except Exception as e:
                progress.write(
                    f"[ERROR] Fallo al decodificar base64 para text_index={text_index}: {e}"
                )
                progress.update(1)
                continue

            for aug_idx in range(n_aug_per_sample):
                aug_img, aug_meta = _random_image_augmentation(img)
                aug_b64 = _image_to_base64(aug_img)
                aug_text = augment_text(text, aug_idx)

                aug_entry = {
                    "original_index": i,
                    "original_text_index": text_index,
                    "original_image_id": match.get("image_id"),

                    "text": aug_text,
                    "image_base64": aug_b64,

                    "image_metadata": match.get("image_metadata"),
                    "distance": match.get("distance"),

                    "is_augmented": True,
                    "augmentation_index": aug_idx,
                    "augmentation_metadata": aug_meta,
                }
                all_samples.append(aug_entry)

            progress.update(1)

    # 2) Guardar local
    if save_local:
        LOCAL_AUGMENTED_JSON.parent.mkdir(parents=True, exist_ok=True)

        with LOCAL_AUGMENTED_JSON.open("w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        print(f"JSON guardado en local: {LOCAL_AUGMENTED_JSON}")

    # 3) Guardar en MinIO
    if save_to_minio:
        client = get_minio_client()
        try:
            payload = json.dumps(
                all_samples, ensure_ascii=False, indent=2
            ).encode("utf-8")
            client.put_object(
                config.AUGMENTATION_BUCKET,
                AUGMENTED_JSON_KEY,
                io.BytesIO(payload),
                length=len(payload),
                content_type="application/json",
            )
            print(f"JSON subido a MinIO: {config.AUGMENTATION_BUCKET}/{AUGMENTED_JSON_KEY}")

        except S3Error as e:
            raise RuntimeError(f"Error subiendo JSON de augmentation a MinIO: {e}")

    return all_samples


if __name__ == "__main__":
    samples = augment_dataset(
        n_aug_per_sample=N_AUG_PER_SAMPLE,
        save_local=False,
        save_to_minio=True,
    )
    print(f"Total muestras generadas: {len(samples)}")
