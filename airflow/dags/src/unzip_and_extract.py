import re
from zipfile import ZipFile
from google.cloud import storage
from datetime import datetime, timedelta
import os
from io import BytesIO
import boto3
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
import pandas as pd
import pendulum
from airflow.operators.python import PythonOperator
from airflow import DAG
import logging
from airflow.exceptions import AirflowFailException

task_logger = logging.getLogger("airflow_task")


# Task 3: Extract the zip file to CSV
def extract_zip_to_csv(latest_filename):
        try:  
            with ZipFile("./dummy.zip", 'r') as zipf:
                task_logger.info("Reading csv from zip")
                task_logger.info(f"CONTENTS OF ZIPFILE :\n {zipf.namelist()}")

                bytes_data = zipf.read(latest_filename.split(".")[0] + ".csv")
                task_logger.info("Read csv from zip")
                trip_data_df = pd.read_csv(BytesIO(bytes_data))
                #task_logger.info(f"Working directory: {os.getcwd()}")

            os.system('rm -rf ./dummy.zip')
            df_csv_path = "/opt/airflow/dags/data/" + latest_filename.split(".")[0] + ".csv"

            trip_data_df.to_csv(df_csv_path)
            task_logger.info(f"Extracted {latest_filename.split('.')[0]}.csv from .zip")
            return {'df_path': df_csv_path, 'blob_path': f"trip_data/{os.path.basename(df_csv_path)}"}
        except Exception as e:
            task_logger.error(f"Failed to read {latest_filename}; encountering {e}")
            raise AirflowFailException
