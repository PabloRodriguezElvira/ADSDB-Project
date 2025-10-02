from pathlib import Path
from minio.error import S3Error
from src.common.minio_client import get_minio_client
import kagglehub


client = get_minio_client()


def get_json_file_from_folder(folder_path: str, json_file: str):
    file_path = Path(folder_path) / json_file
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró: {file_path}")
    return file_path
    

def upload_file_to_bucket(bucket: str, destination_object: str, json_file: Path):
    client.fput_object(bucket, destination_object, str(json_file), content_type="application/json")
    print(f"[OK] Subido {json_file} a s3://{bucket}/{destination_object}")


def main():
    # Download latest version of the dataset 'epicrecipes'
    path = kagglehub.dataset_download("hugodarwood/epirecipes")


    # Put json file inside the temporal_landing bucket
    bucket = "landing-zone"
    json_file = get_json_file_from_folder(path, "full_format_recipes.json")
    destination_object = f"temporal_landing/{json_file.name}"

    upload_file_to_bucket(bucket, destination_object, json_file)


if __name__ == "__main__":
    try:
        main()
    except S3Error as e:
        print(f"Error MinIO: {e}")




