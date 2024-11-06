import pandas as pd
import numpy as np
import math
import os
import logging
from airflow.exceptions import AirflowSkipException

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Data cleaning functions
def remove_missing_values(df):
    logging.info("Removing missing values.")
    return df.dropna()


def remove_duplicates(df):
    logging.info("Removing duplicate rows.")
    return df.drop_duplicates()


# Data preprocessing helper functions
def data_type_conversion(df):
    logging.info("Converting data types.")
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])
    df["rideable_type"] = df["rideable_type"].astype("category")
    df["member_casual"] = df["member_casual"].astype("category")
    df["start_station_id"] = df["start_station_id"].astype("str")
    df["end_station_id"] = df["end_station_id"].astype("str")
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def extract_temporal_features(df):
    logging.info("Extracting temporal features.")
    df["year"] = df["started_at"].dt.year
    df["month"] = df["started_at"].dt.month
    df["day"] = df["started_at"].dt.day
    df["hour"] = df["started_at"].dt.hour
    df["day_name"] = df["started_at"].dt.day_name()
    df["duration"] = round(
        (df["ended_at"] - df["started_at"]) / pd.Timedelta(minutes=1), 0
    )
    df["distance_km"] = df.apply(
        lambda row: haversine_distance(
            row["start_lat"], row["start_lng"], row["end_lat"], row["end_lng"]
        ),
        axis=1,
    )
    return df


def remove_invalid_data(df):
    df = df[(df["duration"] > 5) & (df["duration"] < 1440)]
    df = df[df["distance_km"] > 0]
    return df


# Main function for data cleaning and preprocessing
def clean_and_preprocess_data(extract_dir, clean_dir, chunk_size=50000, **context):
    # Get execution date to determine the file to process
    execution_date = context["execution_date"]
    year_month = execution_date.strftime("%Y%m")
    filename = f"{year_month}-bluebikes-tripdata.csv"
    file_path = os.path.join(extract_dir, filename)
    output_path = os.path.join(clean_dir, f"preprocessed_{filename}")

    logging.info(f"Starting data cleaning and preprocessing for file: {filename}")

    # Check if the file exists
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist. Skipping processing.")
        return AirflowSkipException

    # Remove the output file if it already exists to avoid appending to old data
    if os.path.exists(output_path):
        logging.warning(f"Output file {output_path} already exists. Hence, removing it....")
        os.remove(output_path)

    # Process data in chunks to handle large files
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        try:
            # Step 1: Clean data
            chunk = remove_missing_values(chunk)
            chunk = remove_duplicates(chunk)
            
            # Step 2: Preprocess data
            chunk = data_type_conversion(chunk)
            chunk = extract_temporal_features(chunk)
            chunk = remove_invalid_data(chunk)
            
            # Write each processed chunk to the output file
            chunk.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        except Exception as e:
            logging.error(f"Error encountered during chunk processing for {file_path} : {e}")
            raise e
        

    
    # Push the processed file name to XCom
    # context['ti'].xcom_push(key='cleaned_file', value=f"preprocessed_{filename}")
    context["ti"].xcom_push(key="cleaned_file", value=output_path)
    logging.info(
        f"Data cleaning and preprocessing completed for file: {filename}. Output saved to: {output_path}"
    )
    logging.info(f"Data cleaning and preprocessing completed for file: {filename}")
