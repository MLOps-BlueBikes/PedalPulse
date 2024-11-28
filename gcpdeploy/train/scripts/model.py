# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib
# import mlflow
# import mlflow.sklearn
# from load_data import load_data_from_gcs
# import os
# from app.config import GCP_BUCKET_NAME, GCP_DATA_FILE, MODEL_SAVE_PATH, LOG_PATH
# import logging
# from scripts.push_model_to_vertex_ai import push_model_to_vertex_ai
# from google.cloud import storage


# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger()

# # Log data loading
# logger.info("Loading data...")

# lr_model_path = os.path.join(MODEL_SAVE_PATH, "linear_regression_model.pkl")
# dt_model_path = os.path.join(MODEL_SAVE_PATH, "decision_tree_model.pkl")



# # Load data from GCS using configuration
# data_cleaned = load_data_from_gcs(GCP_BUCKET_NAME, GCP_DATA_FILE)
# # Data preprocessing

# logger.info("Data loaded. Preprocessing...")
# #data_cleaned['started_at'] = pd.to_datetime(data_cleaned['started_at'], format='mixed', errors='coerce')
# data_cleaned['started_at'] = pd.to_datetime(data_cleaned['started_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# # Drop rows where 'started_at' is NaT
# data_cleaned = data_cleaned.dropna(subset=['started_at'])

# data_cleaned.set_index('started_at', inplace=True)
# logger.info("Data preprocessed. Starting resampling...")

# hourly_station_data = data_cleaned.groupby('start_station_name').resample('h').agg({
#     'month': 'first',                # Keep first value (since it's constant for each hour)
#     'hour': 'first',                 # Same as above
#     'day_name': 'first',             # Same
#     'duration': 'sum',               # Sum durations for the hour
#     'distance_km': 'sum',            # Sum distances for the hour
#     'Temperature (°F)': 'mean',      # Average temperature
#     'Humidity': 'mean',              # Average humidity
#     'Wind Speed': 'mean',            # Average wind speed
#     'Precip.': 'sum',                # Total precipitation for the hour
#     'Condition': 'first',            # Keep first condition as representative
#     'BikeUndocked': 'sum'            # Sum undocked bikes
# })
# # Reset the index to make it easier to work with
# hourly_station_data = hourly_station_data.reset_index()
# # Ensure start_station_name is in DataFrame

# # One-hot encode 'day_name' and 'Condition'
# encoded_data = pd.get_dummies(hourly_station_data, columns=['day_name', 'Condition'], drop_first=True)

# X = encoded_data.drop('BikeUndocked', axis=1)  # Features
# y = encoded_data['BikeUndocked']  # Target
# # Frequency encoding This approach encodes each category by its frequency of occurrence in the dataset.
# X = X.drop(columns=['started_at'])

# logger.info(f"Encoded X columns: {X.columns}")

# station_freq = X['start_station_name'].value_counts().to_dict()
# X['start_station_name'] = X['start_station_name'].map(station_freq).fillna(0)
# X = X.fillna(0)

# # Train, Validation, Test Split (Chronologically)
# train_size = int(0.7 * len(X))  # 70% for training
# val_size = int(0.15 * len(X))   # 15% for validation
# test_size = len(X) - train_size - val_size  # 15% for test
# X_train, y_train = X[:train_size], y[:train_size]
# X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
# X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# logger.info("Data split. Starting model training...")
# #mlflow.set_tracking_uri("http://localhost:5000")
# # Set up local tracking
# mlflow.set_tracking_uri("file:/app/mlruns")

# # Training and Logging with MLflow
# mlflow.set_experiment("Training Experiment")

# # # Linear Regression
# # with mlflow.start_run(run_name="Linear Regression with GridSearch"):
# #     mlflow.log_param("train_size", train_size)
# #     mlflow.log_param("validation_size", val_size)

# #     pipe_lr = Pipeline([('scl', StandardScaler()), ('reg', LinearRegression())])
# #     logger.info("Pipeline created")
# #     param_grid_lr = {'reg__fit_intercept': [True, False], 'reg__positive': [True, False]}
# #     grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5)
# #     logger.info("Starting GridSearchCV for Linear Regression...")
# #     grid_lr.fit(X_train, y_train)

# #     y_pred = grid_lr.predict(X_val)
# #     mse = mean_squared_error(y_val, y_pred)
# #     r2 = r2_score(y_val, y_pred)

# #     logger.info(f"Validation MSE: {mse}, R2: {r2}")

# #     # Log metrics and parameters with MLflow
# #     mlflow.log_params(grid_lr.best_params_)
# #     mlflow.log_metric("mean_squared_error", mse)
# #     mlflow.log_metric("r2_score", r2)

# #     logger.info("Attempting to save models...")
# #     #mlflow.sklearn.log_model(grid_lr.best_estimator_, "linear_regression_model")

# #     #joblib.dump(grid_lr.best_estimator_, "models/model.pkl")
# #     joblib.dump(grid_lr.best_estimator_, lr_model_path)
# #     # Save model to path specified in config

# #Decision Tree
# with mlflow.start_run(run_name="Decision Tree with GridSearch"):
#     mlflow.log_param("train_size", train_size)
#     mlflow.log_param("validation_size", val_size)

#     param_grid_dt = {
#         'max_depth': [3, 5, 10, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     logger.info("Starting GridSearchCV for Decision Trees...")
#     grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=param_grid_dt, cv=5)
#     grid_dt.fit(X_train, y_train)

#     y_pred = grid_dt.predict(X_val)
#     mse = mean_squared_error(y_val, y_pred)
#     r2 = r2_score(y_val, y_pred)

#     mlflow.log_params(grid_dt.best_params_)
#     mlflow.log_metric("mean_squared_error", mse)
#     mlflow.log_metric("r2_score", r2)

#     # Save model to path specified in config
#     joblib.dump(grid_dt.best_estimator_, dt_model_path)

# # Final model evaluation on the test set
# logger.info("Final model evaluation on the test set...")

# # Linear Regression test evaluation

# # lr_best_model = joblib.load(lr_model_path)
# # y_test_pred_lr = lr_best_model.predict(X_test)
# # mse_lr = mean_squared_error(y_test, y_test_pred_lr)
# # r2_lr = r2_score(y_test, y_test_pred_lr)
# # logger.info(f"Linear Regression Test MSE: {mse_lr}, R2: {r2_lr}")

# # Decision Tree test evaluation

# dt_best_model = joblib.load(dt_model_path)
# y_test_pred_dt = dt_best_model.predict(X_test)
# mse_dt = mean_squared_error(y_test, y_test_pred_dt)
# r2_dt = r2_score(y_test, y_test_pred_dt)
# logger.info(f"Decision Tree Test MSE: {mse_dt}, R2: {r2_dt}")

# def upload_to_gcs(local_path, bucket_name, destination_blob_name):
#     """Uploads a file to Google Cloud Storage."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(local_path)
#     print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

# # Example usage in training script:
# local_model_path = "/app/models/model.pkl"
# gcs_bucket = os.environ.get("GCS_BUCKET", GCP_BUCKET_NAME)
# gcs_model_path = f"models/model.pkl"

# # Upload model to GCS after training
# upload_to_gcs(local_model_path, GCP_BUCKET_NAME, gcs_model_path)

import logging
import os
import joblib
from google.cloud import storage
import pandas as pd
import gcsfs
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
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
dt_model_path = os.path.join(MODEL_SAVE_PATH, "decision_tree_model.pkl")


# GCS and Environment Configurations
BUCKET_NAME = os.getenv("BUCKET_NAME", "test-blue-bikes")
MODEL_DIR = os.getenv("MODEL_DIR", f"gs://{BUCKET_NAME}/models/model/")
LOG_DIR = os.getenv("LOG_DIR", f"gs://{BUCKET_NAME}/logs/")

fs = gcsfs.GCSFileSystem(project='blue-bike-prediction')
storage_client = storage.Client()


# Upload logs to GCS
def upload_log_to_gcs(log_content, log_path):
    try:
        with fs.open(log_path, 'wb') as log_file:
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
        df['started_at'] = pd.to_datetime(df['started_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # Drop rows where 'started_at' is NaT
        df = df.dropna(subset=['started_at'])
        
        # Set the datetime column as the index for resampling
        df.set_index('started_at', inplace=True)
        logger.info("Data preprocessed. Starting resampling...")

        # Resample data by hour for each station
        hourly_station_data = df.groupby('start_station_name').resample('h').agg({
            'month': 'first',                # Keep first value (since it's constant for each hour)
            'hour': 'first',                 # Same as above
            'day_name': 'first',             # Same
            'duration': 'sum',               # Sum durations for the hour
            'distance_km': 'sum',            # Sum distances for the hour
            'Temperature (°F)': 'mean',      # Average temperature
            'Humidity': 'mean',              # Average humidity
            'Wind Speed': 'mean',            # Average wind speed
            'Precip.': 'sum',                # Total precipitation for the hour
            'Condition': 'first',            # Keep first condition as representative
            'BikeUndocked': 'sum'            # Sum undocked bikes
        }).reset_index()

        logging.info("Data resampling completed.")
        # Prepare features (X) and target (y)
        X = hourly_station_data.drop(columns=['BikeUndocked'])
        y = hourly_station_data['BikeUndocked']

        # One-hot encode categorical variables
        logger.info("Applying one-hot encoding...")
        X = pd.get_dummies(X, columns=['day_name', 'Condition'], drop_first=True)

        # Frequency encode the 'start_station_name' column
        logger.info("Applying frequency encoding to 'start_station_name'...")
        station_freq = X['start_station_name'].value_counts().to_dict()
        X['start_station_name'] = X['start_station_name'].map(station_freq).fillna(0)

        # Drop unnecessary columns
        logger.info("Dropping unnecessary columns...")
        X = X.drop(columns=['started_at'])

        # Fill any remaining missing values with 0
        X = X.fillna(0)

        logger.info(f"Final feature columns: {X.columns}")

        # Split the data chronologically into train, validation, and test sets
        train_size = int(0.7 * len(X))  # 70% for training
        val_size = int(0.15 * len(X))   # 15% for validation
        test_size = len(X) - train_size - val_size  # 15% for test

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        logger.info("Data splitting into train, validation, and test sets completed.")
        # Upload log after preprocessing
        upload_log_to_gcs("Data preprocessing and resampling completed.", LOG_DIR + "/preprocessing_log.txt")

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise
# Train the model
def train_model(X_train, y_train, X_val, y_val):
    try:
        # Define and train a Decision Tree Regressor
        model = DecisionTreeRegressor(random_state=42)
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Evaluate the model
        y_pred = grid_search.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        logger.info(f"Validation MSE: {mse}, R2: {r2}")
        # Upload log after training
        upload_log_to_gcs(f"Training completed. Validation MSE: {mse}, R2: {r2}.", LOG_DIR + "/training_log.txt")
        
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise


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
        df = load_data_from_gcs(GCP_BUCKET_NAME, GCP_DATA_FILE)
        #data_cleaned = load_data_from_gcs(GCP_BUCKET_NAME, GCP_DATA_FILE)

        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

        # Train the model
        model = train_model(X_train, y_train, X_val, y_val)

        # Save artifacts
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        
        # Example usage in training script:
        joblib.dump(model, dt_model_path)
        local_model_path = "/app/models/decision_tree_model.pkl"
    
        gcs_model_path = f"models/model/model.pkl"

        upload_to_gcs(local_model_path, GCP_BUCKET_NAME, gcs_model_path)

    except Exception as e:
        logger.error(f"failed: {e}")
        raise


if __name__ == "__main__":
    main()
