# Import necessary libraries
import os
import pandas as pd
from io import StringIO
from google.cloud import storage
import tensorflow_data_validation as tfdv
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Load the Google Cloud Storage client using the credentials file directly
client = storage.Client.from_service_account_json(
    "./pedalpulse-440019-919eead68e28 (1).json"
)


# Define a function to fetch data from GCP bucket
def read_from_gcp_bucket(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    df = pd.read_csv(StringIO(content))
    return df


# Define Pandera schema for data validation
schema = DataFrameSchema(
    {
        "ride_id": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "rideable_type": Column(
            pa.String,
            checks=Check.isin(["classic_bike", "electric_bike"]),
            nullable=False,
        ),
        "started_at": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "ended_at": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "start_station_name": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "start_station_id": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "end_station_name": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "end_station_id": Column(
            pa.String, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "start_lat": Column(
            pa.Float, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "start_lng": Column(
            pa.Float, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "end_lat": Column(
            pa.Float, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "end_lng": Column(
            pa.Float, checks=Check(lambda x: x.notnull()), nullable=False
        ),
        "member_casual": Column(
            pa.String, checks=Check.isin(["casual", "member"]), nullable=False
        ),
    }
)


# Function to validate schema using Pandera
def validate_schema(df):
    try:
        schema.validate(df)
        print("Schema validation passed.")
    except pa.errors.SchemaError as e:
        print("Schema validation failed:", e)


# Function to process data and validate schema for each month
def process_monthly_data(bucket_name, start_month, end_month):
    for month in pd.date_range(start=start_month, end=end_month, freq="MS").strftime(
        "%Y%m"
    ):
        print(f"\nProcessing data for {month}...")

        # Load data from GCP
        blob_name = f"{month}-bluebikes-tripdata.csv"
        df = read_from_gcp_bucket(bucket_name, blob_name)

        # Validate data with Pandera
        validate_schema(df)

        # Generate and save statistics with TFDV
        statistics = tfdv.generate_statistics_from_dataframe(df)
        tfdv.write_stats_text(statistics, f"statistics_{month}.pbtxt")

        # Infer schema from data and save it
        schema = tfdv.infer_schema(statistics)
        tfdv.write_schema_text(schema, f"schema_{month}.pbtxt")

        print(f"Statistics and schema saved for {month}.")


# Main function to run the script
def main():
    bucket_name = "raw_data_bucket_pedal_pulse"
    start_month = "2023-12"
    end_month = "2024-09"
    process_monthly_data(bucket_name, start_month, end_month)


# Execute the main function
if __name__ == "__main__":
    main()
