import logging
import time
import os
import ast
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_to_gcs(bucket_name, file_paths):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Convert file_paths from string to list if needed
    if isinstance(file_paths, str):
        try:
            file_paths = ast.literal_eval(file_paths)
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing file paths: {e}")
            return []

    for file_path in file_paths:
        blob_name = os.path.basename(file_path)
        blob = bucket.blob(blob_name)

        # Check if file exists before attempting upload
        if not os.path.exists(file_path):
            logging.warning(f"File {file_path} does not exist. Skipping upload.")
            continue

        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            try:
                blob.upload_from_filename(file_path)
                logging.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
                break
            except Exception as e:
                logging.error(f"Error uploading {file_path}: {e}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff
                if retry_count == max_retries:
                    logging.error(f"Failed to upload {file_path} after {max_retries} retries")

