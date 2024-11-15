
import re
from zipfile import ZipFile
from google.cloud import storage
from datetime import datetime, timedelta
import os
import boto3
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
import pandas as pd
import pendulum
from airflow.operators.python import PythonOperator
from airflow import DAG
import logging
from airflow.exceptions import AirflowFailException

task_logger = logging.getLogger("airflow_task")


def connect_to_bucket():

    os.environ['AWS_ACCESS_KEY_ID']='AKIAQR5EPL346VTJOGQV'
    os.environ['AWS_SECRET_ACCESS_KEY']='+NOCUNyebHsQFr0I0LNBP1UJPXnFV63MLUbZghIN'
    bucket_name = 'hubway-data'

    # Create a session using your AWS credentials
    session = boto3.Session()
    s3 = session.resource('s3')

    # Specify the bucket name
    try:
        bucket = s3.Bucket(bucket_name)
        task_logger.info("hubway-data S3 bucket accessed")
    except Exception as e:
        task_logger.error(f'Failed to access bucket: {bucket_name}')
        raise AirflowFailException(f'Failure due to access bucket: {bucket_name}\n {e}')
    return bucket_name  



    # Task 2: Get the trip dataset from the S3 bucket
def get_trip_dataset(bucket_name, fetch_date=None, ds=None):
        
        os.environ['AWS_ACCESS_KEY_ID']='AKIAQR5EPL346VTJOGQV'
        os.environ['AWS_SECRET_ACCESS_KEY']='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'
        session = boto3.Session()
        s3 = session.resource('s3')
        bucket = s3.Bucket(bucket_name)

        if not fetch_date:
            date_obj = datetime.fromisoformat(ds) - timedelta(days=30)

        else:
            date_obj = datetime.fromisoformat(fetch_date) - timedelta(days=30)
            task_logger.info(f'----------DS VARIABLE IS : {fetch_date}  ')

        
        yr_mnth = date_obj.strftime('%Y%m')
        task_logger.info(f"Fetching dataset for {yr_mnth}")
        
        logging.info(f"Looking for {yr_mnth} tripdata")
        latest_filename = ''
        try:
            for obj in bucket.objects.all():
                if re.match(f'{yr_mnth}.+(?:-tripdata.zip)$', obj._key):
                    latest_filename = obj._key
                    break 
        except Exception as e:
            task_logger.exception(e)

        if latest_filename == '':
            raise AirflowFailException(f"Dataset for year-month {yr_mnth} not found")
        
        task_logger.info(f"Selected {latest_filename} for extraction from S3")
                
        bucket.download_file(Key=latest_filename, Filename="./dummy.zip")
        task_logger.info("Successfully downloaded .zip of dataset")
        
        return latest_filename