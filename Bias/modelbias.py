import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from load_data import load_data_from_gcs, load_data_with_gcsfs
import os

# Import configuration values
from app.config import GCP_BUCKET_NAME, GCP_DATA_FILE, MODEL_SAVE_PATH, LOG_PATH
import logging

lr_model_path = os.path.join(MODEL_SAVE_PATH, "linear_regression_model.pkl")
dt_model_path = os.path.join(MODEL_SAVE_PATH, "decision_tree_model.pkl")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

bucket_name = "trip_data_bucket_testing"
data_path = "trip_data_bucket_testing/Final-data/data_cleaned.csv"

# Load data from GCS using configuration
data_cleaned = load_data_from_gcs(
    GCP_BUCKET_NAME, GCP_DATA_FILE
)  # Replace with your actual file path

# Data preprocessing
logger.info("Preprocessing data...")
data_cleaned["started_at"] = pd.to_datetime(
    data_cleaned["started_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)
data_cleaned = data_cleaned.dropna(subset=["started_at"])
data_cleaned.set_index("started_at", inplace=True)

# Resample data hourly
logger.info("Resampling data...")
hourly_station_data = (
    data_cleaned.groupby("start_station_name")
    .resample("h")
    .agg(
        {
            "month": "first",
            "hour": "first",
            "day_name": "first",
            "duration": "sum",
            "distance_km": "sum",
            "Temperature (°F)": "mean",
            "Humidity": "mean",
            "Wind Speed": "mean",
            "Precip.": "sum",
            "Condition": "first",
            "BikeUndocked": "sum",
            "rideable_type": "first",  # Capture the rideable_type (classic bike / e bike)
        }
    )
    .reset_index()
)

# One-hot encode categorical features
logger.info("Encoding categorical features...")
encoded_data = pd.get_dummies(
    hourly_station_data,
    columns=["day_name", "Condition", "rideable_type"],
    drop_first=True,
)

# Prepare features and target
X = encoded_data.drop(["BikeUndocked", "started_at"], axis=1)
y = encoded_data["BikeUndocked"]
station_freq = X["start_station_name"].value_counts().to_dict()
X["start_station_name"] = X["start_station_name"].map(station_freq).fillna(0)
X = X.fillna(0)

# Split data into training and test sets
logger.info("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple Linear Regression model for demonstration
pipe_lr = Pipeline([("scl", StandardScaler()), ("reg", LinearRegression())])
pipe_lr.fit(X_train, y_train)

# Predict and evaluate on the entire test set
y_pred = pipe_lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logger.info(f"Overall Linear Regression MSE: {mse}, R2: {r2}")

# Bias Analysis: Slicing by most and least frequent stations
logger.info("Performing bias analysis on station frequency...")
most_frequent_station = (
    station_freq.idxmax()
)  # Get the station name with the maximum frequency
least_frequent_station = (
    station_freq.idxmin()
)  # Get the station name with the minimum frequency

# Slicing the test set for most frequent station
X_test_most = X_test[X_test["start_station_name"] == most_frequent_station]
y_test_most = y_test[X_test["start_station_name"] == most_frequent_station]

# Slicing the test set for least frequent station
X_test_least = X_test[X_test["start_station_name"] == least_frequent_station]
y_test_least = y_test[X_test["start_station_name"] == least_frequent_station]

# Evaluate on slices
y_pred_most = pipe_lr.predict(X_test_most)
y_pred_least = pipe_lr.predict(X_test_least)

mse_most = mean_squared_error(y_test_most, y_pred_most)
r2_most = r2_score(y_test_most, y_pred_most)

mse_least = mean_squared_error(y_test_least, y_pred_least)
r2_least = r2_score(y_test_least, y_pred_least)

logger.info(f"Most frequent station - MSE: {mse_most}, R2: {r2_most}")
logger.info(f"Least frequent station - MSE: {mse_least}, R2: {r2_least}")

# Bias Analysis: Slicing by rideable_type
logger.info("Performing bias analysis on rideable_type...")
# Slice data by rideable_type (classic bike vs. e bike)
X_test_classic = X_test[X_test["rideable_type_classic bike"] == 1]
y_test_classic = y_test[X_test["rideable_type_classic bike"] == 1]

X_test_ebike = X_test[X_test["rideable_type_e bike"] == 1]
y_test_ebike = y_test[X_test["rideable_type_e bike"] == 1]

# Evaluate on slices
y_pred_classic = pipe_lr.predict(X_test_classic)
y_pred_ebike = pipe_lr.predict(X_test_ebike)

mse_classic = mean_squared_error(y_test_classic, y_pred_classic)
r2_classic = r2_score(y_test_classic, y_pred_classic)

mse_ebike = mean_squared_error(y_test_ebike, y_pred_ebike)
r2_ebike = r2_score(y_test_ebike, y_pred_ebike)

logger.info(f"Classic Bike - MSE: {mse_classic}, R2: {r2_classic}")
logger.info(f"E-Bike - MSE: {mse_ebike}, R2: {r2_ebike}")

# Generate visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing MSE comparison across station frequency and rideable type
sns.barplot(
    x=[
        "Overall",
        "Most Frequent Station",
        "Least Frequent Station",
        "Classic Bike",
        "E-Bike",
    ],
    y=[mse, mse_most, mse_least, mse_classic, mse_ebike],
)
plt.title("MSE Comparison Across Data Slices")
plt.ylabel("Mean Squared Error")
plt.show()

logger.info("Bias analysis and visualizations completed.")
