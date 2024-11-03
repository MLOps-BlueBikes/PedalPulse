from google.cloud import storage
import os
import pandas as pd
from io import StringIO
from data_cleaning import remove_missing_values, remove_duplicates
from preprocess_ride_data import data_type_conversion, extract_temporal_features, haversine_distance, remove_invalid_data

# Set up the credentials if not already configured
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./data/pedal-pulse-raw-data-5b8626b891ce.json"

def read_from_gcp_bucket(bucket_name, blob_name):
    # Initialize a storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) from the bucket
    blob = bucket.blob(blob_name)

    # Download the contents of the blob as a string
    content = blob.download_as_text()
    df = pd.read_csv(StringIO(content))
    
    return df

# Example usage
bucket_name = "raw_data_bucket_pedal_pulse"
blob_name = "202401-bluebikes-tripdata.csv"

# Read content from the GCP bucket
df = read_from_gcp_bucket(bucket_name, blob_name)

print(df.head())

df = remove_missing_values(df)
df = remove_duplicates(df)

df = data_type_conversion(df)

df = extract_temporal_features(df)

print(df.columns)

df = remove_invalid_data(df)
