from io import BytesIO
import re
from zipfile import ZipFile
from google.cloud import storage
from datetime import datetime, timedelta
import os
import boto3
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
import pandas as pd
import pendulum
from airflow.operators.python_operator import PythonOperator
from airflow.decorators import task,dag
import logging
from airflow.exceptions import AirflowFailException

task_logger= logging.getLogger("airflow_task")

@dag(
    dag_id="monthly_tripdata_dag",
    schedule_interval="0 0 7 * *",
    start_date=pendulum.datetime(2023, 9, 1, tz="UTC"),
    description="DAG to process Bluebikes trip data",
    catchup=True,
    default_args={'email':["mlopsgcpproject@gmail.com"],
                    'email_on_failure':True,
                    'email_on_retry':True}
)


def bb_trips_data():



    @task
    def connect_to_bucket():
        #os.environ['AWS_ACCESS_KEY_ID']='use your access keyid'
        #os.environ['AWS_SECRET_ACCESS_KEY']='use your access key'
      
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
    
    @task
    def get_trip_dataset(bucket_name,fetch_date=None,ds=None):
        #os.environ['AWS_ACCESS_KEY_ID']='use your access keyid'
        #os.environ['AWS_SECRET_ACCESS_KEY']='use your access key'
        session = boto3.Session()
        s3 = session.resource('s3')
        bucket = s3.Bucket(bucket_name)
        if not fetch_date:
            date_obj = datetime.fromisoformat(ds)-timedelta(days=30)
        # Format to 'YYYYMM'
        else:
             date_obj = datetime.fromisoformat(fetch_date)-timedelta(days=30)
        yr_mnth = date_obj.strftime('%Y%m')
        task_logger.info(f"Fetching dataset for {yr_mnth}")
        
        logging.info(f"Looking for {yr_mnth} tripdata")
        latest_filename=''
        try:
            for obj in bucket.objects.all():
                if re.match(f'{yr_mnth}.+(?:-tripdata.zip)$',obj._key):
                        latest_filename=obj._key 
                        break 
        except Exception as e:
            task_logger.exception(e)

        if latest_filename=='':
             raise AirflowFailException(f"Dataset for year-month {yr_mnth} not found")
        
        task_logger.info(f"Selected {latest_filename} for extraction from S3")
                
        bucket_name = 'hubway-data'
 
        s3.Bucket(bucket_name).download_file(Key=latest_filename,Filename="./dummy.zip")

        task_logger.info("Succesfully downloaded .zip of dataset")
        
        return latest_filename

        
        
    @task(task_id="extract_to_csv")     
    def  extract_zip_to_csv(latest_filename): 
        try:  
            with ZipFile("./dummy.zip",'r') as zip:
                task_logger.info("Reading csv from zip")
                bytes_data=zip.read(latest_filename.split(".")[0]+".csv")
                task_logger.info("Read csv from zip")
                trip_data_df = pd.read_csv(BytesIO(bytes_data))
                task_logger.info(f"Im in {os.getcwd()}")

            os.system('rf -rm ./dummy.zip')
            df_csv_path="/opt/airflow/dags/data/"+latest_filename.split(".")[0]+".csv"

            trip_data_df.to_csv(df_csv_path)
            task_logger.info(f"Extracted {latest_filename.split('.')[0]}.csv from .zip")
            return {'df_path':df_csv_path,'blob_path':f"trip_data/{os.path.basename(df_csv_path)}"} 
        except Exception as e:
             task_logger.error(f"Failed to read {latest_filename}; encountering {e}")
             raise AirflowFailException

    
    def push_to_gcp_bucket(destination_blob_path,source_path,**kwargs):
            
            gcp_bucket_name="trip_data_bucket_testing"
            task_logger.info(f'Pushing {os.path.basename(source_path)} to GCP Bucket')
            task_logger.info(f"Attempting to access bucket- {gcp_bucket_name}")
            try:
                
                upload_task = LocalFilesystemToGCSOperator(
                task_id=f'upload_to_gcs',
                src=source_path,
                dst=destination_blob_path,
                bucket=gcp_bucket_name,
                gcp_conn_id='gcp-connection'                                      )
                

                upload_task.execute(context=kwargs)
                task_logger.info(f'Pushed {os.path.basename(source_path)} to GCP Bucket')
                os.system(f'rm -rf {source_path}')
            except Exception as e :
                    task_logger.error("Failed to access bucket due to following error: \n {e}")
                    raise AirflowFailException
                 
   
  
        
    
    upload_to_gcs= PythonOperator(
                            task_id='upload_to_gcs',
                            python_callable=push_to_gcp_bucket,
                            op_kwargs={'destination_blob_path':'{{ task_instance.xcom_pull(task_ids="extract_to_csv", key="return_value")["blob_path"] }}',
                                 'source_path':'{{ task_instance.xcom_pull(task_ids="extract_to_csv", key="return_value")["df_path"] }}'
                            },
                            provide_context=True,
                        )

    
    s3_bucket=connect_to_bucket()
    tripdata_filename=get_trip_dataset(s3_bucket,fetch_date='2024-11-03T15:25:27.444')
    extract_zip_to_csv(tripdata_filename) >> upload_to_gcs


bb_trips_data() 





