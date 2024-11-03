import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import resample
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
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
    # Step 1: Load Data
    df = load_data('your_data.csv')
    if df is None:
        return

    # Define columns
    label_col = 'true_label'
    pred_col = 'predicted_label'
    sensitive_feature = 'gender'
    group_to_balance = 'female'  # Example underrepresented group

    # Step 2: Detect Bias
    metrics_before = detect_bias(df, label_col, pred_col, sensitive_feature)

    # Step 3: Mitigate Bias
    balanced_df = mitigate_bias(df, sensitive_feature, group_to_balance)

    # Step 4: Re-evaluate Metrics after Mitigation
    metrics_after = detect_bias(balanced_df, label_col, pred_col, sensitive_feature)

    # Step 5: Document the Process
    document_process(metrics_before, metrics_after)

if __name__ == "__main__":
    main()
