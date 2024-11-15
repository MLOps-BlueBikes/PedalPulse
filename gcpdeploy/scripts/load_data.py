from google.cloud import storage
import pandas as pd
import requests
import gcsfs
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



# def load_data_with_gcsfs(bucket_name, data_path):
#     """Loads data from GCS using gcsfs."""
#     file_path = f"gs://{bucket_name}/{data_path}"
#     data = pd.read_csv(file_path, storage_options={"token": "cloud"})
#     return data


def load_data_with_gcsfs(bucket_name, data_path):
    """Loads data from GCS using gcsfs with service account credentials."""
    file_path = f"gs://{bucket_name}/{data_path}"
    data = pd.read_csv(file_path, storage_options={"token": "cloud"})
    return data

