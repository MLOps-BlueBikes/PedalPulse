import logging
import time
import os
import ast
from google.cloud import storage
from airflow.exceptions import AirflowFailException


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_to_gcs(bucket_name, file_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob_name = os.path.basename(file_path)
    blob = bucket.blob(blob_name)

    # Check if file exists before attempting upload
    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist. Skipping upload.")
        #continue

    retry_count = 0
    max_retries = 10
    while retry_count < max_retries:
        try:
            blob.upload_from_filename(file_path)
            logging.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
            break
        except Exception as e:
            retry_count += 1
            time.sleep(2 ** retry_count)  # Exponential backoff
            if retry_count == max_retries:
                logging.error(f"Failed to upload {file_path} after {max_retries} retries")
            logging.error(f"Error uploading {file_path}: {e}")
            raise AirflowFailException(f'Failed to upload {file_path}:{e}')
