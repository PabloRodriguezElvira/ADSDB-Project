import os
import io
import argparse
from PIL import Image
import boto3
from botocore.client import Config

def s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://localhost:9000'),
        aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'admin'),
        aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'admin123'),
        config=Config(signature_version='s3v4')
    )

def list_objects(s3, bucket, prefix):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in resp.get('Contents', []):
        yield obj['Key']

def convert_to_png(data, size=(512, 512)):
    img = Image.open(io.BytesIO(data))

    img.thumbnail(size)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-bucket', required=True)
    parser.add_argument('--dst-bucket', required=True)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--size', nargs=2, type=int, default=(512, 512))
    args = parser.parse_args()

    s3 = s3_client()

    for key in list_objects(s3, args.src_bucket, args.prefix):
        print("Processing:", key)
        obj = s3.get_object(Bucket=args.src_bucket, Key=key)
        data = obj['Body'].read()

        try:
            png_bytes = convert_to_png(data, size=tuple(args.size))
        except Exception as e:
            print("Error with", key, ":", e)
            continue

        base, _ = os.path.splitext(key)
        dst_key = base + ".png"

        s3.put_object(Bucket=args.dst_bucket, Key=dst_key, Body=png_bytes)
        print("Saving in:", args.dst_bucket, "/", dst_key)

if __name__ == "__main__":
    main()
