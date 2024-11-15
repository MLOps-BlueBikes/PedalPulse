import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from load_data import load_data_from_gcs,load_data_with_gcsfs
import os
# Import configuration values
from app.config import GCP_BUCKET_NAME, GCP_DATA_FILE, MODEL_SAVE_PATH, LOG_PATH
import logging
os.makedirs("/app/mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:/app/mlruns")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Log data loading
logger.info("Loading data...")

lr_model_path = os.path.join(MODEL_SAVE_PATH, "linear_regression_model.pkl")
dt_model_path = os.path.join(MODEL_SAVE_PATH, "decision_tree_model.pkl")


# Load data from GCS
bucket_name = "trip_data_bucket_testing"
data_path = "trip_data_bucket_testing/Final-data/data_cleaned.csv"

# Load data from GCS using configuration
data_cleaned = load_data_from_gcs(GCP_BUCKET_NAME, GCP_DATA_FILE)
# Usage


# Data preprocessing

logger.info("Data loaded. Preprocessing...")
#data_cleaned['started_at'] = pd.to_datetime(data_cleaned['started_at'], format='mixed', errors='coerce')
data_cleaned['started_at'] = pd.to_datetime(data_cleaned['started_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Drop rows where 'started_at' is NaT
data_cleaned = data_cleaned.dropna(subset=['started_at'])

data_cleaned.set_index('started_at', inplace=True)
logger.info("Data preprocessed. Starting resampling...")

hourly_station_data = data_cleaned.groupby('start_station_name').resample('h').agg({
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
})
# Reset the index to make it easier to work with
hourly_station_data = hourly_station_data.reset_index()
# Ensure start_station_name is in DataFrame

# One-hot encode 'day_name' and 'Condition'
encoded_data = pd.get_dummies(hourly_station_data, columns=['day_name', 'Condition'], drop_first=True)

X = encoded_data.drop('BikeUndocked', axis=1)  # Features
y = encoded_data['BikeUndocked']  # Target
# Frequency encoding This approach encodes each category by its frequency of occurrence in the dataset.
X = X.drop(columns=['started_at'])

logger.info(f"Encoded X columns: {X.columns}")

station_freq = X['start_station_name'].value_counts().to_dict()
X['start_station_name'] = X['start_station_name'].map(station_freq).fillna(0)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info("Data split. Starting model training...")


# Training and Logging with MLflow
#mlflow.set_experiment("Training Experiment")
experiment = mlflow.get_experiment_by_name("Training Experiment")
if experiment is None:
    experiment_id = mlflow.create_experiment("Training Experiment")
    logger.info(f"Created experiment with ID: {experiment_id}")
else:
    logger.info(f"Using existing experiment: {experiment.experiment_id}")

# Linear Regression
with mlflow.start_run(run_name="Linear Regression with GridSearch"):
    pipe_lr = Pipeline([('scl', StandardScaler()), ('reg', LinearRegression())])
    logger.info("Pipeline created")
    param_grid_lr = {'reg__fit_intercept': [True, False], 'reg__positive': [True, False]}
    grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5)
    logger.info("Starting GridSearchCV for Linear Regression...")
    grid_lr.fit(X_train, y_train)

    y_pred = grid_lr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Linear Regression MSE: {mse}, R2: {r2}")
    logger.info("Attempting to save models...")
    # Log metrics and save model
    mlflow.log_params(grid_lr.best_params_)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)
    # Save model to path specified in config
    joblib.dump(grid_lr.best_estimator_, lr_model_path)


#Decision Tree
with mlflow.start_run(run_name="Decision Tree with GridSearch"):
    param_grid_dt = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=param_grid_dt, cv=5)
    logger.info("Starting GridSearchCV for Decision trees...")
    grid_dt.fit(X_train, y_train)

    y_pred = grid_dt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Decision Tree MSE: {mse}, R2: {r2}")
    mlflow.log_params(grid_dt.best_params_)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Save model to path specified in config
    joblib.dump(grid_dt.best_estimator_, dt_model_path)

print(f"Training completed. Models saved to {MODEL_SAVE_PATH}.")