from minio import Minio

def get_minio_client():
    """
    Connects to the MinIO client using the credentials specified in docker-compose.yaml
    """
    return Minio(
        "localhost:9000",
        access_key="admin",
        secret_key="admin123",
        secure=False
    )