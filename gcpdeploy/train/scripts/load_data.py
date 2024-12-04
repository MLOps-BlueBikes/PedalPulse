from google.cloud import storage
import pandas as pd
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "scripts/key.json"


def load_data_from_gcs(bucket_name, file_path):
    """
    Load CSV data from Google Cloud Storage.

    Args:
        bucket_name (str): GCS bucket name.
        file_path (str): Path to the CSV file in the bucket.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return pd.read_csv(blob.open("rt"))
