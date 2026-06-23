"""
Upload Model Artifacts to S3
===============================
Run this AFTER training (train_model.py) and AFTER configuring your AWS
credentials (see deployment steps in README.md).

Requires: pip install boto3
"""

import boto3
import os
from botocore.exceptions import ClientError, NoCredentialsError

# ---- EDIT THESE ----
BUCKET_NAME = "your-house-price-bucket"   # must be globally unique on AWS
REGION = "ap-south-1"                      # change to your preferred region
LOCAL_FILES = [
    "models/house_price_model.pkl",
    "models/scaler.pkl",
    "models/metadata.json",
    "data/housing.csv",
]
S3_PREFIX = "house-price-model/"           # "folder" inside the bucket
# ---------------------


def create_bucket_if_not_exists(s3_client, bucket_name, region):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError:
        print(f"Creating bucket '{bucket_name}' in {region}...")
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print("Bucket created.")


def upload_files(s3_client, bucket_name, files, prefix):
    for local_path in files:
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        key = prefix + os.path.basename(local_path)
        s3_client.upload_file(local_path, bucket_name, key)
        print(f"  Uploaded: {local_path}  ->  s3://{bucket_name}/{key}")


if __name__ == "__main__":
    try:
        s3 = boto3.client("s3", region_name=REGION)
        create_bucket_if_not_exists(s3, BUCKET_NAME, REGION)
        print("\nUploading files...")
        upload_files(s3, BUCKET_NAME, LOCAL_FILES, S3_PREFIX)
        print("\nDone. Files are now in S3.")
        print(f"Browse at: https://s3.console.aws.amazon.com/s3/buckets/{BUCKET_NAME}")
    except NoCredentialsError:
        print("ERROR: AWS credentials not found.")
        print("Run 'aws configure' first, or set environment variables:")
        print("  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION")
    except ClientError as e:
        print(f"AWS error: {e}")
