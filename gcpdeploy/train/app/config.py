import os

GCP_BUCKET_NAME = "blue-bikes"
GCP_DATA_FILE = "data_cleaned.csv"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # gcpdeploy directory
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")

LOG_PATH = "logs/training.log"

# import os
# GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
