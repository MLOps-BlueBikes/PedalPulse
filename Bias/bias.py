import os
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import resample
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
def detect_bias(df, label_col, pred_col, sensitive_feature):
    logging.info(f"Detecting bias for sensitive feature: {sensitive_feature}")
    metrics_frame = MetricFrame(
        metrics={'accuracy': accuracy_score, 'recall': recall_score, 'precision': precision_score},
        y_true=df[label_col],
        y_pred=df[pred_col],
        sensitive_features=df[sensitive_feature]
    )
    logging.info("Metrics by group:")
    logging.info(metrics_frame.by_group)
    return metrics_frame.by_group

# Function to mitigate bias by resampling underrepresented groups
def mitigate_bias(df, sensitive_feature, group_to_balance):
    logging.info(f"Mitigating bias for group: {group_to_balance}")
    majority_group = df[df[sensitive_feature] != group_to_balance]
    minority_group = df[df[sensitive_feature] == group_to_balance]

    # Resample minority group to match majority group size
    minority_resampled = resample(minority_group,
                                  replace=True,
                                  n_samples=len(majority_group),
                                  random_state=1)
    balanced_df = pd.concat([majority_group, minority_resampled])
    
    logging.info("Bias mitigation by resampling completed.")
    return balanced_df

# Function to document the bias detection and mitigation steps
def document_process(metrics_before, metrics_after, output_file="bias_mitigation_log.txt"):
    with open(output_file, "w") as f:
        f.write("Bias Detection and Mitigation Report\n")
        f.write("="*40 + "\n\n")
        f.write("Metrics Before Mitigation:\n")
        f.write(str(metrics_before) + "\n\n")
        
        f.write("Metrics After Mitigation:\n")
        f.write(str(metrics_after) + "\n\n")
        
        f.write("Bias Mitigation Technique: Resampling underrepresented group\n")
        f.write("Note: Additional mitigation techniques may be required depending on the observed performance.\n")
    
    logging.info("Documentation saved to bias_mitigation_log.txt")

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
    sensitive_feature = 'gender'
    group_to_balance = 'female'  # Example underrepresented group

    # Detect bias
    metrics_before = detect_bias(df, label_col, pred_col, sensitive_feature)

    # Mitigate bias
    balanced_df = mitigate_bias(df, sensitive_feature, group_to_balance)

    # Re-evaluate metrics after mitigation
    metrics_after = detect_bias(balanced_df, label_col, pred_col, sensitive_feature)

    # Document the process
    document_process(metrics_before, metrics_after)

if __name__ == "__main__":
    main()
