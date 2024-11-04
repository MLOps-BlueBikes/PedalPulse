from airflow import DAG
from airflow.operators.python import PythonOperator
import os
from datetime import datetime, timedelta, timezone
from src.ingest_data import ingest_data
from src.unzip_file import unzip_file
from src.upload_to_gcp import upload_to_gcs
from src.track_with_dvc import track_with_dvc
from airflow.exceptions import AirflowFailException

# Enable pickle support for XCom, allowing complex data to be passed between tasks
download_dir = '/opt/airflow/downloads'
extract_dir = '/opt/airflow/extracted_files'
# output_dir = '/opt/airflow/output'
# output_file = f'{output_dir}/bluebikes-tripdata.csv'

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)
#os.makedirs(os.path.dirname(output_file), exist_ok=True)


# Define default arguments for the DAG
default_args = {
    'owner': 'Muskan',
    'start_date': datetime(2023, 10, 1),  # Adjust as needed
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=10),  # Delay before retries
    'email':["muskankh03@gmail.com"],
    'email_on_failure':True
}

# Create a DAG instance
dag = DAG(
    'bluebikes_data_pipeline',
    default_args=default_args,
    description='DAG for Blue Bikes Prediction Project',
    schedule_interval='@monthly',  
    catchup=True,
)

# Generate URLs dynamically
base_url = "https://s3.amazonaws.com/hubway-data"

def get_monthly_url(logical_date):
    """
    Generates the URL for the data file based on the logical date (execution date).
    """
    try:
        year = logical_date.year
        month = logical_date.month
        return f"{base_url}/{year}{month:02d}-bluebikes-tripdata.zip"
    except Exception as e:
        raise AirflowFailException(f"Failed to obtain dataset urls: {e}")
        
    
# Define the task to generate the monthly URL
get_monthly_url_task = PythonOperator(
    task_id='get_monthly_url_task',
    python_callable=lambda **context: [get_monthly_url(context['logical_date'])],
    provide_context=True,
    dag=dag
)
# Adjusted ingest_data_task to pass execution_date as a datetime object
ingest_data_task = PythonOperator(
    task_id='ingest_data_task',
    python_callable=ingest_data,
    op_kwargs={
        'urls': "{{ task_instance.xcom_pull(task_ids='get_monthly_url_task') }}",
        'download_dir': download_dir
    },
    provide_context=True,  # Enable context passing
    dag=dag
)


unzip_file_task = PythonOperator(
    task_id='unzip_file_task',
    python_callable=unzip_file,
    op_kwargs={
        'zip_paths': "{{ task_instance.xcom_pull(task_ids='ingest_data_task') }}",
        'extract_to': extract_dir,
    },
    dag=dag
)


bucket_name = 'bluebikes_bucket_before_preprocessing'

# Define the upload task
upload_to_gcs_task = PythonOperator(
    task_id='upload_to_gcs_task',
    python_callable=upload_to_gcs,
    op_kwargs={
        'bucket_name': bucket_name,
        'file_paths': "{{ task_instance.xcom_pull(task_ids='unzip_file_task') }}"
    },
    dag=dag
)
# dvc_track_task = PythonOperator(
#     task_id='dvc_track_task',
#     python_callable=track_with_dvc,
#     op_kwargs={'file_path': extract_dir},
#     dag=dag
# )


# Set task dependencies
get_monthly_url_task >> ingest_data_task >> unzip_file_task >> upload_to_gcs_task 
#>> dvc_track_task

if __name__ == "__main__":
    dag.cli()
