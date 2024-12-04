from flask import Flask, jsonify, request
import os
import joblib
import pandas as pd
from google.cloud import storage

app = Flask(__name__)

# Constants
PROJECT_ID = "blue-bike-prediction"
MODEL_FILE_NAME = "model.pkl"
BUCKET_NAME = "test-blue-bikes"
BLOB_NAME = f"models/model/{MODEL_FILE_NAME}"

# Initialize Google Cloud Storage client and download model
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(BLOB_NAME)

# Download the model file from GCS
print(f"Downloading model from: {BLOB_NAME}")
blob.download_to_filename(MODEL_FILE_NAME)

# Load the model
print("Loading model...")
model = joblib.load(MODEL_FILE_NAME)


# Health check endpoint
@app.route(os.environ.get("AIP_HEALTH_ROUTE", "/health"), methods=["GET"])
def health_check():
    return {"status": "healthy"}


# Prediction endpoint
@app.route(os.environ.get("AIP_PREDICT_ROUTE", "/predict"), methods=["POST"])
def predict():
    try:
        # Parse request data
        request_json = request.get_json()
        request_instances = request_json.get("instances", [])

        # Convert input data to DataFrame for prediction
        input_data = pd.DataFrame(request_instances)

        # Ensure the input data matches the model's expected features
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
            input_data = input_data[expected_features]

        # Perform prediction
        predictions = model.predict(input_data)

        # Convert predictions to a list for JSON serialization
        predictions = predictions.tolist()

        # Create response
        response = {"predictions": predictions}
        return jsonify(response)

    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
