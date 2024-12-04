from airflow import DAG
from airflow.operators.python import PythonOperator
import os
from datetime import datetime, timedelta

from src.weather import (
    scrape_multiple_days,
    read_bike_trip_data,
    match_rides_with_weather,
    upload_to_gcp,
)

# Define paths
weather_dir = "/opt/airflow/weather_file"
os.makedirs(weather_dir, exist_ok=True)  # Ensure the directory is created

# Define default arguments for the DAG
default_args = {
    "owner": "Ritika",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
    "catchup": False,
}

# Create a DAG instance
dag = DAG(
    "weather_data_pipeline",
    default_args=default_args,
    description="DAG for integrating weather data with bike trip data",
    catchup=False,
)


def get_month_year(logical_date):
    # Get the year and month from logical_date
    year = logical_date.year
    month = logical_date.month

    # Handle transition from January to December
    if month == 1:
        year -= 1
        month = 12
    else:
        month -= 1

    # Format Jan 2024: 202401
    date = f"{year}{month:02d}"
    return date


get_month_task = PythonOperator(
    task_id="get_month_task",
    python_callable=lambda **context: [get_month_year(context["logical_date"])],
    provide_context=True,
    dag=dag,
)

# Task to read bike trip data from GCP
read_bike_data_task = PythonOperator(
    task_id="read_bike_data_task",
    python_callable=read_bike_trip_data,
    op_kwargs={"month": "{{ task_instance.xcom_pull(task_ids='get_month_task') }}"},
    dag=dag,
)

# Task to scrape weather data
scrape_weather_data_task = PythonOperator(
    task_id="scrape_weather_data_task",
    python_callable=scrape_multiple_days,
    op_args=[read_bike_data_task.output],
    dag=dag,
)

# Task to match weather data with bike trip data
match_bike_weather_task = PythonOperator(
    task_id="match_bike_weather_task",
    python_callable=match_rides_with_weather,
    op_args=[scrape_weather_data_task.output],
    dag=dag,
)

# Task to upload data to GCS
upload_to_gcs_task = PythonOperator(
    task_id="upload_to_gcs_task",
    python_callable=upload_to_gcp,
    op_kwargs={
        "file_path": match_bike_weather_task.output,
        "bucket_name": "trip_data_bucket_testing",
        "month": "202409",
    },
    dag=dag,
)

# Set task dependencies
(
    get_month_task
    >> read_bike_data_task
    >> scrape_weather_data_task
    >> match_bike_weather_task
    >> upload_to_gcs_task
)
