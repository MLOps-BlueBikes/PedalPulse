import os
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score
from google.cloud import storage
import logging

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./data/pedal-pulse-raw-data-5b8626b891ce.json"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Function to detect bias using Fairlearn's MetricFrame
def detect_bias(df, label_col, pred_col, sensitive_features):
    for feature in sensitive_features:
        logging.info(f"Detecting bias for sensitive feature: {feature}")
        metrics_frame = MetricFrame(
            metrics={'accuracy': accuracy_score, 'recall': recall_score, 'precision': precision_score},
            y_true=df[label_col],
            y_pred=df[pred_col],
            sensitive_features=df[feature]
        )
        logging.info(f"Metrics by group for {feature}:")
        logging.info(metrics_frame.by_group)
        print(f"Metrics by group for {feature}:\n{metrics_frame.by_group}")

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
    label_col = 'true_label'
    pred_col = 'predicted_label'
    sensitive_features = ['member_casual', 'rideable_type']

    # Detect bias
    detect_bias(df, label_col, pred_col, sensitive_features)

if __name__ == "__main__":
    main()
