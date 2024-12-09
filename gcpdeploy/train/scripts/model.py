from datetime import datetime
import logging
import os
import sys

sys.path.append("/home/jupyter")
import smtplib
from email.mime.text import MIMEText

import mlflow
from mlflow.exceptions import RestException
import joblib
from google.cloud import storage
import pandas as pd
import gcsfs
from sklearn.linear_model import LinearRegression
from load_data import load_data_from_gcs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
from app.config import GCP_BUCKET_NAME, GCP_DATA_FILE, MODEL_SAVE_PATH, LOG_PATH
from google.cloud import storage

# Import functions to report metrics to GCP
from report_metric import report_model_metric

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
dt_model_path = os.path.join(MODEL_SAVE_PATH, "decision_tree_model.pkl")
lr_model_path = os.path.join(MODEL_SAVE_PATH, "linear_regression_model.pkl")
mlflow.set_tracking_uri("http://35.192.77.77:5000")
mlflow.set_experiment("PedalPulse-Bluebikes")
mlflow_client = mlflow.MlflowClient(tracking_uri="http://35.192.77.77:5000")

# Email settings
EMAIL_SENDER = "athanishreyashri@gmail.com"
EMAIL_PASSWORD = "hqmd vcuf rodd iniu"  # Replace with your email app-specific password
EMAIL_RECIPIENT = "athanishreyashri@gmail.com"

# GCS and Environment Configurations
BUCKET_NAME = os.getenv("BUCKET_NAME", "blue-bikes")
MODEL_DIR = os.getenv("MODEL_DIR", f"gs://{BUCKET_NAME}/models/model/")
LOG_DIR = os.getenv("LOG_DIR", f"gs://{BUCKET_NAME}/logs/")
PROJECT_ID = os.getenv("PROJECT_ID", "bluebike-443722")

fs = gcsfs.GCSFileSystem(project="bluebike-443722")
storage_client = storage.Client()


# Helper function to send email notifications
def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECIPIENT

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


# Upload logs to GCS
def upload_log_to_gcs(log_content, log_path):
    try:
        with fs.open(log_path, "wb") as log_file:
            log_file.write(log_content.encode())
        logger.info(f"Logs uploaded to {log_path}")
    except Exception as e:
        logger.error(f"Failed to upload logs: {e}")
        raise


def preprocess_data(df):
    """
    Preprocess the data:
    - Resample the data hourly by start_station_name.
    - Encode categorical variables.
    - Scale numerical features.

    Args:
    df (pandas.DataFrame): Raw input DataFrame.

    Returns:
    tuple: Preprocessed features (X), target (y), and the preprocessor pipeline.
    """
    try:
        df["started_at"] = pd.to_datetime(
            df["started_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

        # Drop rows where 'started_at' is NaT
        df = df.dropna(subset=["started_at"])

        # Set the datetime column as the index for resampling
        df.set_index("started_at", inplace=True)
        logger.info("Data preprocessed. Starting resampling...")

        # Resample data by hour for each station
        hourly_station_data = (
            df.groupby("start_station_name")
            .resample("h")
            .agg(
                {
                    "month": "first",  # Keep first value (since it's constant for each hour)
                    "hour": "first",  # Same as above
                    "day_name": "first",  # Same
                    "duration": "sum",  # Sum durations for the hour
                    "distance_km": "sum",  # Sum distances for the hour
                    "Temperature (°F)": "mean",  # Average temperature
                    "Humidity": "mean",  # Average humidity
                    "Wind Speed": "mean",  # Average wind speed
                    "Precip.": "sum",  # Total precipitation for the hour
                    "Condition": "first",  # Keep first condition as representative
                    "BikeUndocked": "sum",  # Sum undocked bikes
                }
            )
            .reset_index()
        )

        logging.info("Data resampling completed.")
        # Prepare features (X) and target (y)
        X = hourly_station_data.drop(columns=["BikeUndocked"])
        y = hourly_station_data["BikeUndocked"]

        # One-hot encode categorical variables
        logger.info("Applying one-hot encoding...")
        X = pd.get_dummies(X, columns=["day_name", "Condition"], drop_first=True)

        # Frequency encode the 'start_station_name' column
        logger.info("Applying frequency encoding to 'start_station_name'...")
        station_freq = X["start_station_name"].value_counts().to_dict()
        X["start_station_name"] = X["start_station_name"].map(station_freq).fillna(0)

        # Drop unnecessary columns
        logger.info("Dropping unnecessary columns...")
        X = X.drop(columns=["started_at"])

        # Fill any remaining missing values with 0
        X = X.fillna(0)

        logger.info(f"Final feature columns: {X.columns}")

        # Split the data chronologically into train, validation, and test sets
        train_size = int(0.7 * len(X))  # 70% for training
        val_size = int(0.15 * len(X))  # 15% for validation
        test_size = len(X) - train_size - val_size  # 15% for test

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = (
            X[train_size : train_size + val_size],
            y[train_size : train_size + val_size],
        )
        X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

        logger.info("Data splitting into train, validation, and test sets completed.")
        # Upload log after preprocessing
        upload_log_to_gcs(
            "Data preprocessing and resampling completed.",
            LOG_DIR + "/preprocessing_log.txt",
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise


# Train the model
def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    lr_run_id = ""
    dt_run_id = ""
    date = datetime.today()
    try:
        with mlflow.start_run(run_name=f"run_{date}"):
            mlflow.log_param("dataset_version", date.strftime("%m-%Y"))
            mlflow.log_param("train_size", "0.7")
            mlflow.log_param("val_size", "0.15")
            # # # Linear Regression
            with mlflow.start_run(
                run_name="Linear Regression with GridSearch", nested=True
            ) as lr_run:

                lr_run_id = lr_run.info.run_id
                pipe_lr = Pipeline(
                    [("scl", StandardScaler()), ("reg", LinearRegression())]
                )
                logger.info("Pipeline created")
                param_grid_lr = {
                    "reg__fit_intercept": [True, False],
                    "reg__positive": [True, False],
                }
                grid_search_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5)
                logger.info("Starting GridSearchCV for Linear Regression...")
                grid_search_lr.fit(X_train, y_train)

                y_pred = grid_search_lr.predict(X_val)
                lr_val_mse = round(mean_squared_error(y_val, y_pred), 3)
                lr_val_r2 = round(r2_score(y_val, y_pred), 3)

                y_pred = grid_search_lr.predict(X_test)
                lr_test_mse = round(mean_squared_error(y_test, y_pred), 3)
                lr_test_r2 = round(r2_score(y_test, y_pred), 3)

                logger.info(f"Validation MSE: {lr_val_mse}, R2: {lr_val_r2}")
                logger.info(f"Test MSE: {lr_test_mse}, R2: {lr_test_r2}")

                # Log metrics and parameters with MLflow
                mlflow.log_params(grid_search_lr.best_params_)
                mlflow.log_metric("test_mean_squared_error", lr_test_mse)
                mlflow.log_metric("test_r2_score", lr_test_r2)
                mlflow.log_metric("val_mean_squared_error", lr_val_mse)
                mlflow.log_metric("val_r2_score", lr_val_r2)

                logger.info("Attempting to save models...")
                # mlflow.sklearn.log_model(grid_lr.best_estimator_, "linear_regression_model")

                # joblib.dump(grid_lr.best_estimator_, "models/model.pkl")
                joblib.dump(grid_search_lr.best_estimator_, lr_model_path)
                # Save model to path specified in config
            ### Decision Tree
            with mlflow.start_run(
                run_name="Decision Tree Regressor with GridSearch", nested=True
            ) as dt_run:

                dt_run_id = dt_run.info.run_id
                # Define and train a Decision Tree Regressor
                model = DecisionTreeRegressor(random_state=42)
                param_grid = {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}
                grid_search_dt = GridSearchCV(model, param_grid, cv=5)

                logger.info("Starting GridSearchCV for Decision Tree...")
                grid_search_dt.fit(X_train, y_train)

                # Evaluate the model
                y_pred = grid_search_dt.predict(X_val)
                dt_val_mse = round(mean_squared_error(y_val, y_pred), 3)
                dt_val_r2 = round(r2_score(y_val, y_pred), 3)

                y_pred = grid_search_dt.predict(X_test)
                dt_test_mse = round(mean_squared_error(y_test, y_pred), 3)
                dt_test_r2 = round(r2_score(y_test, y_pred), 3)

                mlflow.log_params(grid_search_dt.best_params_)
                mlflow.log_metric("test_mean_squared_error", dt_test_mse)
                mlflow.log_metric("test_r2_score", dt_test_r2)
                mlflow.log_metric("val_mean_squared_error", dt_val_mse)
                mlflow.log_metric("val_r2_score", dt_val_r2)

                joblib.dump(grid_search_dt.best_estimator_, dt_model_path)
                logger.info(f"Validation MSE: {dt_val_mse}, R2: {dt_val_r2}")
                logger.info(f"Test MSE: {dt_test_mse}, R2: {dt_test_r2}")

            bm = pick_best_model(test_x=X_test, test_y=y_test)
            best_model = joblib.load(lr_model_path if bm == "lr" else dt_model_path)
            best_model_r2 = lr_val_r2 if bm == "lr" else dt_val_r2
            best_model_mse = lr_val_mse if bm == "lr" else dt_val_mse
            best_model_run_id = lr_run_id if bm == "lr" else dt_run_id

            # Report RMSE and MSE metrics to cloud monitoring
            report_model_metric(PROJECT_ID, best_model_r2, 'rmse')
            report_model_metric(PROJECT_ID, best_model_mse, 'mse')

            # Upload log after training
            # upload_log_to_gcs(f"Training completed. Validation MSE: {mse}, R2: {r2}.", LOG_DIR + "/training_log.txt")
            print(f"Selected {bm}")
            try:
                latest_model_version = mlflow_client.get_latest_versions(
                    "pedalpulse_demand_prediction_model"
                )

            except RestException as e:
                with mlflow.start_run(run_id=best_model_run_id, nested=True):
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="pedalpulse_models",
                        registered_model_name="pedalpulse_demand_prediction_model",
                    )
                    return best_model

            if latest_model_version:
                lm = latest_model_version[0]
                print(f"Latest model is {lm._name}-{lm._version} ")
                print(f"Run id {lm.run_id}")
                lm_run_metrics = mlflow_client.get_run(run_id=lm.run_id).data.metrics

                print(
                    f"""Latest model performance : {lm_run_metrics}
                        Current trained best model performance: mse={best_model_mse}, r2={best_model_r2}"""
                )
                if best_model_r2 > lm_run_metrics["val_r2_score"]:
                    print(
                        "The newly trained model is performing better than latest model!!!, registering it"
                    )
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="pedalpulse_models",
                        registered_model_name="pedalpulse_demand_prediction_model",
                    )
                    return best_model
                else:
                    print(
                        "The newly trained model is underperforming compared to production model, skipping registration"
                    )
                    return None
            else:
                return None

    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise


def pick_best_model(test_x, test_y):
    try:

        df = pd.concat([test_x, test_y], axis=1)

        lr_model = joblib.load(lr_model_path)
        dt_model = joblib.load(dt_model_path)

        lr_r2 = round(r2_score(lr_model.predict(test_x), test_y), 3)
        dt_r2 = round(r2_score(dt_model.predict(test_x), test_y), 3)

        lr_mse = round(mean_squared_error(lr_model.predict(test_x), test_y), 3)
        dt_mse = round(mean_squared_error(dt_model.predict(test_x), test_y), 3)

        station_freq = df["start_station_name"].value_counts()
        # Get the most and least frequent stations
        most_frequent_stations = station_freq.head(10).index
        least_frequent_stations = station_freq.tail(10).index

        # Filter data for most and least frequent stations
        most_freq_data = df[df["start_station_name"].isin(most_frequent_stations)]
        least_freq_data = df[df["start_station_name"].isin(least_frequent_stations)]

        lr_r2_most_freq = round(
            r2_score(
                lr_model.predict(most_freq_data.drop(["BikeUndocked"], axis=1)),
                most_freq_data["BikeUndocked"],
            ),
            3,
        )
        lr_r2_least_freq = round(
            r2_score(
                lr_model.predict(least_freq_data.drop(["BikeUndocked"], axis=1)),
                least_freq_data["BikeUndocked"],
            ),
            3,
        )
        logger.info(
            f"Linear Regression R2 score for top 10 most frequent stations:{lr_r2_most_freq} least frequent stations: {lr_r2_least_freq}"
        )
        dt_r2_most_freq = round(
            r2_score(
                dt_model.predict(most_freq_data.drop(["BikeUndocked"], axis=1)),
                most_freq_data["BikeUndocked"],
            ),
            3,
        )
        dt_r2_least_freq = round(
            r2_score(
                dt_model.predict(least_freq_data.drop(["BikeUndocked"], axis=1)),
                least_freq_data["BikeUndocked"],
            ),
            3,
        )
        logger.info(
            f"Decision Tree Regression R2 score for top 10 most frequent stations:{dt_r2_most_freq} least frequent stations:         {dt_r2_least_freq}"
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        send_email(
            subject="Pipeline Notification: Failure",
            body=f"The pipeline failed with the following error:\n\n{e}",
        )

    if abs(lr_r2_most_freq - lr_r2_least_freq) < abs(
        dt_r2_most_freq - dt_r2_least_freq
    ):

        model = "lr"
    elif dt_r2 > lr_r2:
        model = "dt"
    else:
        model = "lr"

    val_mse = dt_mse if model == "dt" else lr_mse
    val_r2 = dt_r2 if model == "dt" else lr_r2
    send_email(
        subject="Pipeline Notification: Model Training Completed",
        body=f"""
            Model training completed successfully.
            Validation MSE: {val_mse}, R2: {val_r2}
            Model uploaded to GCS.
            The selected model is {"Linear Regression" if model=="lr" else "Decision Tree"}
            """,
    )

    return model


def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")


# Main pipeline
def main():
    try:
        # Load data
        print("Loading dataset from GCS")
        df = load_data_from_gcs(GCP_BUCKET_NAME, GCP_DATA_FILE)
        # data_cleaned = load_data_from_gcs(GCP_BUCKET_NAME, GCP_DATA_FILE)
        print("Loaded Dataset. Preprocessing in progress")
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

        # Train the model
        model = train_model(X_train, y_train, X_val, y_val, X_test, y_test)

        if model:
            # Save artifacts
            model_path = os.path.join(MODEL_DIR, "model.pkl")

            # Example usage in training script:
            joblib.dump(model, dt_model_path)
            local_model_path = "./models/decision_tree_model.pkl"

            gcs_model_path = f"models/model/model.pkl"

            upload_to_gcs(local_model_path, GCP_BUCKET_NAME, gcs_model_path)

    except Exception as e:
        logger.error(f"failed: {e}")


if __name__ == "__main__":
    main()
