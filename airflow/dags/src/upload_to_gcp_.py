import logging
import time
import os
import ast
from airflow.exceptions import AirflowSkipException

from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def upload_to_gcs_(bucket_name, file_path):
    client = storage.Client()
    logging.info(f"Trying to fetch gcs bucket:{bucket_name}")
    try:
      bucket = client.get_bucket(bucket_name)
    except Exception as e:
        logging.error(f"Failed to retrieve GCS Bucket: {bucket_name}")
        raise e 
    # blob_name = os.path.basename(file_path)
    # blob = bucket.blob(blob_name)
    blob_name = f"Preprocessed_data/{os.path.basename(file_path)}"
    blob = bucket.blob(blob_name)
    # Log the file path being uploaded
    logging.info(f"Uploading file: {file_path} to GCS bucket: {bucket_name}")

    # Check if file exists before attempting upload
    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist. Skipping upload.")
        raise AirflowSkipException

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
            time.sleep(2**retry_count)  # Exponential backoff
            if retry_count == max_retries:
                logging.error(f"Failed to upload {file_path} after {max_retries} retries")
                raise e 
            
