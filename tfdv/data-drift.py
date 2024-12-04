import os
os.system('pip install tensorflow_data_validation')
!pip install pandera

import logging
from google.cloud import storage
import pandera as pa
import pandas as pd
from pandera import Column, DataFrameSchema, Check
import tensorflow_data_validation as tfdv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path to your service account JSON key
service_account_file = "elated-life-443222-k6-16b113d408b2.json"

# Bucket and file details
bucket_name = "bluebikes_bucket_preprocessing"
file_name = "data_cleaned.csv"
destination_file = "formBucket_data_cleaned.csv"
new_file = "preprocessed_weather_trip_history.csv"

# Google Cloud Storage download
storage_client = storage.Client.from_service_account_json(service_account_file)
bucket = storage_client.bucket(bucket_name)

# Download training data
blob = bucket.blob(file_name)
blob.download_to_filename(destination_file)
logger.info(f"Training data downloaded to {destination_file}")

# Download incoming data
blob.download_to_filename(new_file)
logger.info(f"Incoming data downloaded to {new_file}")

# Define schema for validation
schema = DataFrameSchema(
    {
        "started_at": Column(pa.String, checks=Check(lambda x: x.notnull()), nullable=False),
        "start_station_name": Column(pa.String, checks=Check(lambda x: x.notnull()), nullable=False),
        "start_station_id": Column(pa.String, checks=Check(lambda x: x.notnull()), nullable=False),
        "month": Column(pa.Int, checks=Check(lambda x: 1 <= x <= 12), nullable=False),
        "hour": Column(pa.Int, checks=Check(lambda x: 0 <= x < 24), nullable=False),
        "day_name": Column(pa.String, checks=Check.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]), nullable=False),
        "duration": Column(pa.Float, checks=Check(lambda x: x > 0), nullable=False),
        "distance_km": Column(pa.Float, checks=Check(lambda x: x > 0), nullable=False),
        "Temperature (°F)": Column(pa.Float, checks=Check(lambda x: x.notnull()), nullable=False),
        "Humidity": Column(pa.Float, checks=Check(lambda x: 0 <= x <= 100), nullable=False),
        "Wind Speed": Column(pa.Float, checks=Check(lambda x: x >= 0), nullable=False),
        "Precip.": Column(pa.Float, checks=Check(lambda x: x >= 0), nullable=False),
        "Condition": Column(pa.String, checks=Check(lambda x: x.notnull()), nullable=False),
        "BikeUndocked": Column(pa.Int, checks=Check(lambda x: x >= 0), nullable=False),
    }
)

# Load datasets
training_data = pd.read_csv(destination_file)
incoming_data = pd.read_csv(new_file)

# Validate schema
def validate_schema(df):
    try:
        schema.validate(df)
        logger.info("Schema validation passed.")
    except pa.errors.SchemaError as e:
        logger.error(f"Schema validation failed: {e}")

validate_schema(training_data)
validate_schema(incoming_data)

# Generate statistics
train_stats = tfdv.generate_statistics_from_dataframe(training_data)
incoming_stats = tfdv.generate_statistics_from_dataframe(incoming_data)

# Infer schema from training data statistics
schema = tfdv.infer_schema(statistics=train_stats)

# Helper functions
def get_feature_stats(stats, feature_name):
    for feature in stats.datasets[0].features:
        if feature.path.step[0] == feature_name:
            return feature
    return None

def check_mean_drift(stats1, stats2, feature_name, threshold=40):
    feature1_stats = get_feature_stats(stats1, feature_name)
    feature2_stats = get_feature_stats(stats2, feature_name)

    if feature1_stats and feature2_stats and feature1_stats.num_stats and feature2_stats.num_stats:
        train_mean = feature1_stats.num_stats.mean
        incoming_mean = feature2_stats.num_stats.mean
        if train_mean != 0:
            percentage_change = abs((incoming_mean - train_mean) / train_mean) * 100
            logger.info(f"Train Mean: {train_mean}, Incoming Mean: {incoming_mean}")
            logger.info(f"Percentage Change in Mean: {percentage_change:.2f}%")
            if percentage_change > threshold:
                logger.warning(f"Data drift detected in `{feature_name}`: Mean changed by more than {threshold}%.")
            else:
                logger.info(f"No significant drift detected in `{feature_name}`.")
        else:
            logger.warning(f"Train mean is zero for `{feature_name}`. Cannot calculate percentage change.")
    else:
        logger.error(f"Feature `{feature_name}` not found or not a numerical feature.")

def compare_feature_distribution(stats1, stats2, feature_name):
    feature1 = get_feature_stats(stats1, feature_name)
    feature2 = get_feature_stats(stats2, feature_name)
    if feature1 and feature2 and feature1.string_stats and feature2.string_stats:
        freq1 = {bucket.label: bucket.sample_count for bucket in feature1.string_stats.rank_histogram.buckets}
        freq2 = {bucket.label: bucket.sample_count for bucket in feature2.string_stats.rank_histogram.buckets}
        significant_changes = []
        all_keys = set(freq1.keys()).union(set(freq2.keys()))
        for key in all_keys:
            freq1_value = freq1.get(key, 0)
            freq2_value = freq2.get(key, 0)
            change = abs(freq1_value - freq2_value)
            significant_changes.append((key, change))
        significant_changes = sorted(significant_changes, key=lambda x: x[1], reverse=True)
        logger.info(f"Top significant changes in `{feature_name}` distribution:")
        for key, change in significant_changes[:5]:
            logger.info(f"Value: {key}, Frequency Change: {change}")
    else:
        logger.error(f"Feature `{feature_name}` not found or has no string_stats.")

# Check for drift
check_mean_drift(train_stats, incoming_stats, "duration", threshold=40)
compare_feature_distribution(train_stats, incoming_stats, "start_station_id")

def compare_numerical_distribution_with_drift_detection(stats1, stats2, feature_name, threshold=40):
    feature1 = get_feature_stats(stats1, feature_name)
    feature2 = get_feature_stats(stats2, feature_name)
    if feature1 and feature2 and feature1.num_stats and feature2.num_stats:
        train_mean = feature1.num_stats.mean
        train_std = feature1.num_stats.std_dev
        incoming_mean = feature2.num_stats.mean
        incoming_std = feature2.num_stats.std_dev
        logger.info(f"Training Data - Mean: {train_mean:.2f}, Std Dev: {train_std:.2f}")
        logger.info(f"Incoming Data - Mean: {incoming_mean:.2f}, Std Dev: {incoming_std:.2f}")
        std_change_percentage = abs((incoming_std - train_std) / train_std) * 100 if train_std != 0 else 0
        mean_change_percentage = abs((incoming_mean - train_mean) / train_mean) * 100 if train_mean != 0 else 0
        logger.info(f"Percentage Change in Std Dev: {std_change_percentage:.2f}%")
        logger.info(f"Percentage Change in Mean: {mean_change_percentage:.2f}%")
        if std_change_percentage > threshold:
            logger.warning(f"Significant change detected in `{feature_name}`: Std Dev change exceeds {threshold}%.")
        if mean_change_percentage > threshold:
            logger.warning(f"Significant change detected in `{feature_name}`: Mean change exceeds {threshold}%.")
    else:
        logger.error(f"Feature `{feature_name}` not found or has no num_stats.")

compare_numerical_distribution_with_drift_detection(train_stats, incoming_stats, feature_name="Temperature (°F)", threshold=40)
