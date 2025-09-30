from minio import Minio

def get_minio_client():
    return Minio(
        "localhost:9000",
        access_key="admin",
        secret_key="admin123",
        secure=False
    )