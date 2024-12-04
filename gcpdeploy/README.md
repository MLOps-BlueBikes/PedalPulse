# **Blue Bikes GCP Model Deployment**

## **Overview**
This folder contains the end-to-end workflow for deploying a machine learning model on Google Cloud Platform (GCP). The project includes:

1. **Training Pipeline**: Code to fetch data, train a machine learning model, and upload it to GCP.
2. **Model Serving**: Hosting the trained model on Vertex AI for predictions.
3. **Streamlit App**: A user-facing application that interacts with the model to provide predictions.
4. **CI/CD Pipelines**: Automating the building, training, and deployment of the model and application using Cloud Build.
5. **Retraining**: Preconfigured trigger-based retraining and redeployment of served model based on performance and bias checks.

---

## **Folder Structure**
```plaintext

├── train/scripts          # Contains scripts for training and saving the model
│   ├── load_data.py       # Loading data from gcp bucket
│   ├── model.py           # Script for initial training/retraining of the model 
│   └── train_deploy.py    # Script for deploying the model to Vertex AI
│
├── serve/                 # Contains the model-serving code (Flask API)
│   ├── app.py             # Code for serving the model via Flask
│   ├── Dockerfile         # Dockerfile for serving the model
│   └── requirements.txt   # Dependencies for serving pipeline
│
├── streamlit/             # Contains the Streamlit app for predictions
│   ├── app.py             # Streamlit app code
│   ├── Dockerfile         # Dockerfile for the Streamlit app
│   └── requirements.txt   # Dependencies for the app
│
└── cloudbuild.yaml        # CI/CD pipeline configuration for Cloud Build

```
---

## **Features**
1. **Model Training**:
   - Trains a DecisionTreeRegressor model to predict the number of bikes based on weather, time, and location data.
   - Uploads the trained model to Google Cloud Storage (GCS).
   - Deploys the trained model to Vertex AI as a new endpoint.

2. **Model Serving**:
   - Uses Flask to serve predictions via a REST API.
   - The model is dynamically downloaded from GCS at startup.

3. **Streamlit App**:
   - Allows users to input weather and location data.
   - Fetches predictions from the Vertex AI endpoint dynamically.

4. **CI/CD Pipeline**:
   - Automates the building, training, and deployment process.
   - Rebuilds and redeploys images when the repository is updated.

5. **Experiment Tracking**:
   - Tracks the regression metrics for training runs using an Mlflow tracking server, hosted on GCP
   - Maintains a log of the production model's training and test performance metrics.

   ![Mlflow sample run](https://github.com/MLOps-BlueBikes/PedalPulse/blob/main/gcpdeploy/imgs/mlflow_runs.png)
---

## **Setup Instructions**

### **Prerequisites**
1. A Google Cloud Platform (GCP) project with the following APIs enabled:
   - Vertex AI APIs
   - Cloud Storage APIs
   - Cloud Build APIs
2. A Google Cloud Storage bucket for storing models.
3. `gcloud` CLI installed and authenticated.
4. Docker installed locally (optional for manual builds).
5. Python 3.9+ installed locally.

---

### **Steps to Set Up the Project**

#### **1. Clone the Repository**
```bash
git clone <repository-url>
cd gcpdeploy
```

#### **2. Set Up Environment Variables**
Modify the following environment variables as needed:
- `PROJECT_ID`: Your GCP project ID.
- `REGION`: GCP region (e.g., `us-central1`).
- `BUCKET_NAME`: GCS bucket for storing the trained model.
#### **3. Enable Notebooks API*
#### **4. Configure Google Cloud**
```bash
gcloud config set project <PROJECT_ID>
gcloud config set compute/region <REGION>
```

#### **5. Set Up Cloud Storage**
Create a Cloud Storage bucket to store models:
```bash
gcloud storage buckets create gs://<BUCKET_NAME>
```

#### **6. Enable Required APIs**
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
```

#### **7. Create model-serving image**
```bash
docker build -t gcr.io/<project_name>/model-server:latest .
docker push gcr.io/<project_name>/model-server:latest
```

---

## **Training the Model**
To train the model locally:
```bash
python train/scripts/model.py
```

To run the training pipeline with Cloud Build:
1. Modify the `cloudbuild.yaml` file to point to your project and bucket.
2. Trigger the build:
   ```bash
   gcloud builds submit --config cloudbuild.yaml .
   ```

---

## **Deploying the Streamlit App**
To deploy the Streamlit app on Cloud Run:
```bash
cd streamlit
gcloud run deploy streamlit-app \
  --image gcr.io/<PROJECT_ID>/streamlit:latest \
  --platform managed \
  --region <REGION> \
  --allow-unauthenticated
```

---

## **Monitoring and Retraining**
1. **Monitor Model Performance**:
   - Cloud Monitoring to track model predictions and performance metrics.
   - Detect data drift and performance decay using tools like TensorFlow Data Validation (TFDV).

2. **Trigger Retraining**:
   - Modify the training script to accept new data.
   - Use a Cloud Build trigger or Cloud Functions to automatically retrain when new data is available in GCS.

3. **Deploy Updated Model**:
   - The CI/CD pipeline in `cloudbuild.yaml` automatically pushes the updated model to Vertex AI.

---

## **CI/CD Pipeline**
### **Training/Retraining and Model Deployment**
The `cloudbuild.yaml` file contains the steps to:
1. Build and push the training Docker image.
2. Train the model and save it to GCS.
3. Deploy the trained model to Vertex AI.

Trigger the pipeline:
```bash
gcloud builds submit --config cloudbuild.yaml .
```
Google Cloud Trigger:

![Github Trigger](https://github.com/MLOps-BlueBikes/PedalPulse/blob/main/gcpdeploy/imgs/gcp_trigger_retraining.png)

This trigger enables the `cloudbuild.yaml` build workflow, which contains model training/retraining and deployment steps.    
Model retraining automatically initiates after a push to the main branch. The process fetches recent data from Google Cloud Storage, trains multiple candidate models, and selects a champion model based on performance metrics and bias indicators. If the new model outperforms the current production model, it is deployed.


