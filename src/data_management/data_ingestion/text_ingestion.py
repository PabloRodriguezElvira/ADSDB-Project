import kagglehub
from pathlib import Path
from src.common.minio_client import get_minio_client


def get_json_file_from_folder(folder_path: str, json_file: str):
    file_path = Path(folder_path) / json_file
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró: {file_path}")
    return file_path
    

def upload_file_to_bucket(bucket_name: str, destination_folder: str, json_file: Path):
    client.fput_object(bucket, destination_folder, str(json_file), content_type="application/json")
    print(f"[OK] Subido {json_file} a s3://{bucket_name}/{destination_folder}")




client = get_minio_client() 

# Download latest version of the dataset 'epicrecipes'
path = kagglehub.dataset_download("hugodarwood/epirecipes")


# Put json file inside the temporal_landing bucket
bucket = "landing-zone"
destination_folder = "temporal_landing/full_format_recipes.json"
json_file = get_json_file_from_folder(path, "full_format_recipes.json")

upload_file_to_bucket(bucket, destination_folder, json_file)




