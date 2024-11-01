from io import BytesIO
import re
from zipfile import ZipFile
from google.cloud import storage
import datetime
import os
import boto3
import pandas as pd
import pendulum
from airflow.decorators import task,dag
 

@dag(
    dag_id="monthly_tripdata_dag",
    schedule_interval="0 0 1 * *",
    start_date=pendulum.datetime(2023, 9, 1, tz="UTC"),
    catchup=True,
)
def bb_trips_data():

    @task
    def connect_to_bucket():
        
        #os.environ['AWS_ACCESS_KEY_ID']= "XXXXXXXXXX"
        #os.environ['AWS_SECRET_ACCESS_KEY']= "XXXXXXXXXX"
        bucket_name = 'hubway-data'

        # Create a session using your AWS credentials
        session = boto3.Session()
        s3 = session.resource('s3')

        # Specify the bucket name
        bucket = s3.Bucket(bucket_name)
        return bucket 
    
    @task
    def get_trip_dataset(fetch_date,bucket):
      
        session = boto3.Session()
        s3 = session.resource('s3')

        yr_mnth=fetch_date.strftime('%Y%m')
        #temp_l=list(filter(rgx_match,[obj._key for obj in bucket.objects.all()]))
        
        for obj in bucket.objects.all():
           if re.match(f'{yr_mnth}.+(?:-tripdata.zip)$',obj._key):
                latest_filename=obj._key 
        
        #latest_filename=sorted(temp_l,key= lambda s:int(s[:6]),reverse=True)[0]
        
        bucket_name = 'hubway-data'
 
        s3.Bucket(bucket_name).download_file(Key=latest_filename,Filename="./dummy.zip")
        
        return latest_filename

        
        
    @task     
    def  extract_zip_to_csv(latest_filename):   
        with ZipFile("./dummy.zip",'r') as zip:
            bytes_data=zip.read(latest_filename.split(".")[0]+".csv")
            trip_data_df = pd.read_csv(BytesIO(bytes_data))
        os.system('rf -rm ./dummy.zip')
        df_csv_path="data/"+latest_filename.split(".")[0]+".csv"
        trip_data_df.to_csv(df_csv_path)

        return df_csv_path 

    
    @task
    def push_df_to_gcp(df_filename):
        
        destination_blob_path=f"trip_data/{df_filename.split('.')[0]}"
        gcp_bucket_name="trip_data_bucket_testing"
        push_to_gcp_bucket(gcp_bucket_name,destination_blob_path,df_filename)
        os.system(f'rf -rm {df_filename}')


    
    s3_bucket=connect_to_bucket()
    tripdata_filename=get_trip_dataset('{{execution_date}}',s3_bucket)
    csv_path=extract_zip_to_csv(tripdata_filename)
    push_df_to_gcp(csv_path)


bb_trips_data() 




def push_to_gcp_bucket(bucket_name,destination_blob,source_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/opt/airflow/keys/pedalpulse-440019-919eead68e28.json"

            storage_client=storage.Client()
            bucket=storage_client.get_bucket(bucket_name)
            blob=bucket.blob(destination_blob)
            blob.upload_from_filename(source_path)

            print(f"Dataset {source_path} uploaded to GCP bucket: {bucket_name} at destination {destination_blob}")

'''
def rgx_match(fpath):
  regex_string=r'.+(?:-tripdata.zip)$'
  return re.match(regex_string,fpath)
'''