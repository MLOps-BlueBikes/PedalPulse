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
from scripts.push_model_to_vertex_ai import push_model_to_vertex_ai

os.makedirs("/app/mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:/app/mlruns")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Log data loading
logger.info("Loading data...")

lr_model_path = os.path.join(MODEL_SAVE_PATH, "linear_regression_model.pkl")
dt_model_path = os.path.join(MODEL_SAVE_PATH, "decision_tree_model.pkl")



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

# Train, Validation, Test Split (Chronologically)
train_size = int(0.7 * len(X))  # 70% for training
val_size = int(0.15 * len(X))   # 15% for validation
test_size = len(X) - train_size - val_size  # 15% for test
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

logger.info("Data split. Starting model training...")

# Training and Logging with MLflow
mlflow.set_experiment("Training Experiment")

# Linear Regression
with mlflow.start_run(run_name="Linear Regression with GridSearch"):
    pipe_lr = Pipeline([('scl', StandardScaler()), ('reg', LinearRegression())])
    logger.info("Pipeline created")
    param_grid_lr = {'reg__fit_intercept': [True, False], 'reg__positive': [True, False]}
    grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, cv=5)
    logger.info("Starting GridSearchCV for Linear Regression...")
    grid_lr.fit(X_train, y_train)

    y_pred = grid_lr.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    logger.info(f"Validation MSE: {mse}, R2: {r2}")

    # Log metrics and parameters with MLflow
    mlflow.log_params(grid_lr.best_params_)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    logger.info("Attempting to save models...")
    joblib.dump(grid_lr.best_estimator_, "models/model.pkl")
    # Save model to path specified in config

#Decision Tree
with mlflow.start_run(run_name="Decision Tree with GridSearch"):
    param_grid_dt = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=param_grid_dt, cv=5)
    grid_dt.fit(X_train, y_train)

    y_pred = grid_dt.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    mlflow.log_params(grid_dt.best_params_)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Save model to path specified in config
    joblib.dump(grid_dt.best_estimator_, dt_model_path)

# Final model evaluation on the test set
logger.info("Final model evaluation on the test set...")

# Linear Regression test evaluation

lr_best_model = joblib.load(lr_model_path)
y_test_pred_lr = lr_best_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_test_pred_lr)
r2_lr = r2_score(y_test, y_test_pred_lr)
logger.info(f"Linear Regression Test MSE: {mse_lr}, R2: {r2_lr}")

# Decision Tree test evaluation

dt_best_model = joblib.load(dt_model_path)
y_test_pred_dt = dt_best_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_test_pred_dt)
r2_dt = r2_score(y_test, y_test_pred_dt)
logger.info(f"Decision Tree Test MSE: {mse_dt}, R2: {r2_dt}")

logger.info("Pushing  models to Vertex AI Model Registry...")

# Iterate through all model files in the directory
for model_file in os.listdir(MODEL_SAVE_PATH):
    model_file_path = os.path.join(MODEL_SAVE_PATH, model_file)

    if os.path.isfile(model_file_path):  # Ensure it's a file
        model_name, _ = os.path.splitext(model_file)  # Get model name without extension
        logger.info(f"Pushing model: {model_name}")

        push_model_to_vertex_ai(
            model_path=MODEL_SAVE_PATH,  # Path to the specific model
            project_id="pedalpulse-440019",
            region="us-east1",
            model_display_name=f"{model_name}-vertex-ai"  # Unique display name
        )