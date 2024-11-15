
import re
from zipfile import ZipFile
from google.cloud import storage
from datetime import datetime, timedelta
import os
import boto3
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
import pandas as pd
import pendulum
from airflow.operators.python import PythonOperator
from airflow import DAG
import logging
from airflow.exceptions import AirflowFailException

task_logger = logging.getLogger("airflow_task")

# Task 4: Push the CSV to GCP bucket
def push_to_gcp_bucket(destination_blob_path, source_path, **kwargs):
        gcp_bucket_name = "trip_data_bucket_testing"
        task_logger.info(f'Pushing {os.path.basename(source_path)} to GCP Bucket')
        task_logger.info(f"Attempting to access bucket- {gcp_bucket_name}")
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            try:
                upload_task = LocalFilesystemToGCSOperator(
                    task_id=f'upload_to_gcs',
                    src=source_path,
                    dst=destination_blob_path,
                    bucket=gcp_bucket_name,
                    gcp_conn_id='gcp-connection'
                )
                upload_task.execute(context=kwargs)
                task_logger.info(f'Pushed {os.path.basename(source_path)} to GCP Bucket')
                os.system(f'rm -rf {source_path}')
                return destination_blob_path
                break 
            except Exception as e:
                task_logger.warning(f"Error uploading {source_path}, Retrying......")
                if retry_count==max_retries-1:
                    task_logger.error(f"Failed to access bucket due to following error: \n {e}")
                    raise AirflowFailException
            retry_count+=1
                

def pull_from_gcp_bucket(destination_blob_path, **kwargs):
        gcp_bucket_name = "trip_data_bucket_testing"
        task_logger.info(f'Pulling {os.path.basename(destination_blob_path)} from GCP Bucket')
        task_logger.info(f"Attempting to access bucket- {gcp_bucket_name}")
        retry_count = 0
        max_retries = 10
        local_filepath= '/opt/airflow/dags/data/tripdata/'+os.path.basename(destination_blob_path)
        if not os.path.exists('/opt/airflow/dags/data/tripdata/'):
             os.makedirs('/opt/airflow/dags/data/tripdata/')
        while retry_count < max_retries:
            try:
                upload_task = GCSToLocalFilesystemOperator(
                    task_id=f'upload_to_gcs',
                    filename=local_filepath,
                    object_name=destination_blob_path,
                    bucket=gcp_bucket_name,
                    gcp_conn_id='gcp-connection'
                )
                upload_task.execute(context=kwargs)
                task_logger.info(f'Pulled {os.path.basename(destination_blob_path)} from GCP Bucket and saved to {local_filepath}')
                #os.system(f'rm -rf {source_path}')
                return local_filepath
                break 
            except Exception as e:
                task_logger.warning(f"Error pulling {destination_blob_path} due to {e}, Retrying......")
                if retry_count==max_retries-1:
                    task_logger.error(f"Failed to access bucket due to following error: \n {e}")
                    raise AirflowFailException
            retry_count+=1
                

