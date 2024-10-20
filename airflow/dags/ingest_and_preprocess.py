from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import os
import sys


from src.ingest_data import load_blue_bikes_data, load_weather_data
from src.preprocess_data import preprocess

default_args = {
    'owner': 'divya',
    'start_date': days_ago(1),  # Adjust to your requirement
    'depends_on_past': False,
    'retries': 1
}

dag = DAG(
    'pedal_pulse_ingest_and_preprocess',
    default_args=default_args,
    description='A simple pipeline to ingest and preprocess BlueBikes data',
    schedule_interval='@daily',  # Adjust based on your need
)

PROCESSED_DATA_DIR = 'dags/data'

# Task 1: Ingest Data
def ingest_data():
    # Load BlueBikes data
    blue_bikes_data = load_blue_bikes_data()
    
    # Load Weather data
    weather_data = load_weather_data()

    # Check if the data was loaded successfully
    if blue_bikes_data is not None and weather_data is not None:
        print("Data successfully ingested.")
        return {'blue_bikes': blue_bikes_data, 'weather': weather_data}
    else:
        raise ValueError("Failed to ingest data")
    
# Task 2: Preprocess Data
def preprocess_data(ti):
    # Retrieve the ingested data from the previous task
    data = ti.xcom_pull(task_ids='ingest_data_task')
    blue_bikes_data = data['blue_bikes']
    weather_data = data['weather']
    
    # Call the preprocessing function
    processed_data = preprocess(blue_bikes_data, weather_data)

    # Save the preprocessed data to a file
    output_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
    processed_data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    # Airflow PythonOperators
ingest_data_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data_task',
    python_callable=preprocess_data,
    provide_context=True,  # Allow pulling data between tasks
    dag=dag,
)

# Define the task sequence
ingest_data_task >> preprocess_data_task