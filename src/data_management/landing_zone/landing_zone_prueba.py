import boto3
from botocore.exceptions import NoCredentialsError

# Configurar cliente S3 apuntando a MinIO
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="admin",
    aws_secret_access_key="admin123"
)

# Nombre del bucket
bucket_name = "ejemplo"

# Crear bucket (si no existe)
try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"Bucket '{bucket_name}' creado.")
except s3.exceptions.BucketAlreadyOwnedByYou:
    print(f"Bucket '{bucket_name}' ya existe.")

# Archivo de prueba a subir
archivo_local = "prueba.png"
archivo_remoto = "pruebaArchivoRemoto.png"

# Subir archivo
try:
    s3.upload_file(archivo_local, bucket_name, archivo_remoto)
    print(f"Archivo '{archivo_local}' subido al bucket '{bucket_name}'.")
except FileNotFoundError:
    print(f"El archivo '{archivo_local}' no existe.")
except NoCredentialsError:
    print("Credenciales inválidas.")
