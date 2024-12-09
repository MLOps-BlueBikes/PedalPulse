# **Blue Bikes GCP Model Deployment**

## **Overview**
This folder contains the end-to-end workflow for deploying a machine learning model on Google Cloud Platform (GCP). The project includes:

1. **Training Pipeline**: Code to fetch data, train a machine learning model, and upload it to GCP.
2. **Model Serving**: Hosting the trained model on Vertex AI for predictions.
3. **Streamlit App**: A user-facing application that interacts with the model to provide predictions.
4. **CI/CD Pipelines**: Automating the building, training, and deployment of the model and application using Cloud Build.
5. **Retraining**: Preconfigured trigger-based retraining and redeployment of served model based on performance and bias checks.
6. **Monitoring and Alerts**: Monitoring key model and application metrics and sending alerts for anomalies or degradation in performance.
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

6. **Model Monitoring and Alerts**:
  - Tracking regression metrics for deployed models using Google Cloud Monitoring on Metrics Explorer.
  - Sends alerts through Email if metrics drop below specified threshold.

6. **Hyperparameter Tuning**:
  - The code uses **GridSearchCV**, a systematic search over a predefined hyperparameter grid to evaluate model performance on cross-validated data.
   - Two models are tuned in the pipeline:
     - **Linear Regression:**
       - `fit_intercept`: Whether to calculate the intercept or not.
       - `positive`: Restrict coefficients to positive values.
     - **Decision Tree Regressor:**
       - `max_depth`: The maximum depth of the tree to control overfitting.
       - `min_samples_split`: The minimum number of samples required to split a node.

     **Process of Hyperparameter Tuning:**
   - For each combination of hyperparameters in the grid, the model is trained and validated using 5-fold cross-validation.
   - The model's performance is evaluated using metrics such as **Mean Squared Error (MSE)** and **R² score**.
   - The combination that gives the best performance is selected as the optimal set of hyperparameters.
7. **Model Comparison and Selection Process**
   1. **Training and Validation:**
      - Both models are trained using a 70%-15%-15% split for training, validation, and testing datasets.
      - Hyperparameter tuning is performed using **GridSearchCV** with 5-fold cross-validation.
   
   2. **Station-Based Performance:**
      - R² scores are computed separately for:
        - Top 10 most active stations.
        - Bottom 10 least active stations.
      - This ensures the selected model performs well across diverse station activity levels.
   
   3. **Best Model Selection:**
      - The model with the highest R² score on the **validation data** and the most consistent performance across station categories is chosen.
      - Additionally, the difference in performance between frequent and infrequent stations is analyzed to ensure robustness.
   
   4. **Logging and Tracking:**
      - Performance metrics and best hyperparameters are logged using **MLflow** for reproducibility.
      - The selected model is saved and registered.

---

#### **Results**

- **Linear Regression:**
  - Pros:
    - Simple and interpretable.
    - Works well for linear relationships.
  - Cons:
    - Struggles with complex, non-linear patterns in the data.
  - Performance:
    - Moderate R² score on frequent stations.

- **Decision Tree Regressor:**
  - Pros:
    - Captures non-linear relationships effectively.
    - Handles categorical and numerical data naturally.
  - Cons:
    - Prone to overfitting on training data.
  - Performance:
    - High R² score on frequent stations.
    - Better generalization on infrequent stations compared to Linear Regression.


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
#### **3. Enable Notebooks API**
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

#### **7. Enable Google Cloud Monitoring and Alerts**
```bash
python monitoring/create_mse_metric.py
python monitoring/create_rmse_metric.py
python monitoring/create_rmse_alert.py
```

#### **8. Create model-serving image**
```bash
docker build -t gcr.io/<project_name>/model-serve:latest .
docker push gcr.io/<project_name>/model-serve:latest
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


