# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
from datetime import datetime, timedelta
# from src.blue_bikes_prediction import load_bike_data, preprocess_data
# from src.blue_bikes_prediction import save_to_gcp
from src.ingest_data import ingest_data
from src.unzip_file import unzip_file
from src.load_data import load_data
from src.handle_missing import handle_missing
from src.remove_duplicates import remove_duplicates 

# Enable pickle support for XCom, allowing complex data to be passed between tasks
#conf.set('core', 'enable_xcom_pickling', 'True')
# Define directories for downloads, extraction, and final output
download_dir = '/opt/airflow/downloads'
extract_dir = '/opt/airflow/extracted_files'
output_dir = '/opt/airflow/output'
output_file = f'{output_dir}/bluebikes-tripdata.csv'

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Define default arguments for the DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=10),  # Delay before retries
}



# Create a DAG instance named 'blue_bike_prediction_dag'
dag = DAG(
    'bluebikes_data_pipeline',
    default_args=default_args,
    description='DAG for Blue Bikes Prediction Project',
    schedule_interval='@daily',  # Change as per the project requirements
    catchup=False,
)


# # Task to save preprocessed data to GCP
# save_to_gcp_task = PythonOperator(
#     task_id='save_to_gcp_task',
#     python_callable=save_to_gcp,
#     #op_args=['your_gcp_bucket_name', 'data/processed_data.csv', 'processed_data.csv'],  # Update these arguments
#     op_args=['blue_bikes_bucket', '{{ ti.xcom_pull(task_ids="data_preprocessing_task") }}', 'processed_data.csv'],
  
#     dag=dag,
# )
urls = [
        'https://s3.amazonaws.com/hubway-data/202310-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202311-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202312-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202401-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202402-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202403-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202404-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202405-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202406-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202407-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202408-bluebikes-tripdata.zip', 
        'https://s3.amazonaws.com/hubway-data/202409-bluebikes-tripdata.zip'
        # Add more URLs as needed
    ]
ingest_data_op = PythonOperator(
        task_id='ingest_data_task',
        python_callable=ingest_data,
        op_kwargs={'urls': urls},  # Pass URLs as kwargs
        dag=dag
    )

# Unzip File Task
unzip_file_op = PythonOperator(
    task_id='unzip_file_task',
    python_callable=unzip_file,
    op_kwargs={'zip_paths': [f"{download_dir}/{url.split('/')[-1]}" for url in urls], 'extract_to': extract_dir},
    dag=dag
)


load_data_op = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,  # Call load_data function from load_data module
    op_kwargs={
        'csv_files': [f"{extract_dir}/{url.split('/')[-1].replace('.zip', '.csv')}" for url in urls],
        'output_file': 'combined_data.csv',
        'chunksize': 10000,  # Adjust chunk size if needed
    },
    dag=dag
)




# Remove Duplicates Task
remove_duplicates_task = PythonOperator(
    task_id='remove_duplicates_task',
    python_callable=remove_duplicates,
    op_kwargs={
        'file_path': 'combined_data.csv',  # Use f-string to substitute the variable
        'output_file': f"{output_dir}/final_data.csv"
    },
    dag=dag
)



# Set task dependencies
# Define task dependencies
ingest_data_op >> unzip_file_op >> load_data_op >> remove_duplicates_task

#ingest_data_op >> unzip_file_op >> load_data_op >> remove_duplicates_task>> upload_to_gcp_task >> dvc_track_combined_data >> integrate_weather_data >>dvc_track_final_data

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()





