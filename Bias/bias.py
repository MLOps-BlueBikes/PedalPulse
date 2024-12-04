import os
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score
from google.cloud import storage
import logging

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "./data/pedal-pulse-raw-data-5b8626b891ce.json"
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to fetch data from GCP bucket
def load_data_from_gcp(bucket_name, blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_text()
        df = pd.read_csv(pd.io.common.StringIO(content))
        logging.info("Data loaded successfully from GCP.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from GCP: {e}")
        return None


# Main function
def main():
    # GCP bucket details
    bucket_name = "your_gcp_bucket_name"
    blob_name = "your_data.csv"

    # Load data from GCP
    df = load_data_from_gcp(bucket_name, blob_name)
    if df is None:
        return
    # Define columns
    label_col = "true_label"
    pred_col = "predicted_label"
    sensitive_features = ["member_casual", "rideable_type"]

    # Detect bias
    # detect_bias(df, label_col, pred_col, sensitive_features)


if __name__ == "__main__":
    main()
