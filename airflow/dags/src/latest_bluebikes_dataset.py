import boto3
from zipfile import ZipFile
import os
import pandas as pd
from pathlib import Path
import re
from google.cloud import storage
from io import BytesIO





def connect_to_bucket():
      
    #os.environ['AWS_ACCESS_KEY_ID']= 'XXXXXXXXXX'
    #os.environ['AWS_SECRET_ACCESS_KEY']= 'XXXXXXXXXX'

    gcp_bucket_name="trip_data_bucket_testing"

    # Create a session using your AWS credentials
    session = boto3.Session()
    s3 = session.resource('s3')

    # Specify the bucket name
    bucket_name = 'hubway-data'

    bucket = s3.Bucket(bucket_name)


def get_trip_dataset():
      
    session = boto3.Session()
    s3 = session.resource('s3')

    temp_l=list(filter(rgx_match,[obj._key for obj in bucket.objects.all()]))

    latest_filename=sorted(temp_l,key= lambda s:int(s[:6]),reverse=True)[0]


    s3.Bucket(bucket_name).download_file(Key=latest_filename,Filename="./dummy.zip")

    with ZipFile("./dummy.zip",'r') as zip:
        bytes_data=zip.read(latest_filename.split(".")[0]+".csv")
        trip_data_df = pd.read_csv(BytesIO(bytes_data))
    os.system('rf -rm ./dummy.zip')

    df_csv_path="data/"+latest_filename.split(".")[0]+".csv"
    trip_data_df.to_csv(df_csv_path)
    source_file_path=df_csv_path

def push_df_to_gcp():
    
    destination_blob_path=f"trip_data/{latest_filename.split('.')[0]}"
    push_to_gcp_bucket(gcp_bucket_name,destination_blob_path,source_file_path)
    return {"df":trip_data_df,"path":"data/"+latest_filename.split(".")[0]+".csv"}
    print(trip_data_df.shape)
        

def get_latest_tripdata_df():

    os.environ['AWS_ACCESS_KEY_ID']= 'AKIAQR5EPL34WBH5CLWI'
    os.environ['AWS_SECRET_ACCESS_KEY']= 'dmye2u1P9xJWzURaFRYAqeY1iSFg8qNepK/n+S1/'

    gcp_bucket_name="trip_data_bucket_testing"

    # Create a session using your AWS credentials
    session = boto3.Session()
    s3 = session.resource('s3')

    # Specify the bucket name
    bucket_name = 'hubway-data'

    bucket = s3.Bucket(bucket_name)

    # Create a session using your AWS credentials
    session = boto3.Session()
    s3 = session.resource('s3')

    temp_l=list(filter(rgx_match,[obj._key for obj in bucket.objects.all()]))

    latest_filename=sorted(temp_l,key= lambda s:int(s[:6]),reverse=True)[0]

    zip_file_path="/opt/airflow/dags/data/dummy.zip"
    s3.Bucket(bucket_name).download_file(Key=latest_filename,Filename= zip_file_path)

    with ZipFile(zip_file_path,'r') as zip:
        bytes_data=zip.read(latest_filename.split(".")[0]+".csv")
        trip_data_df = pd.read_csv(BytesIO(bytes_data))
    os.system(f'rm -rf {zip_file_path}')

    df_csv_path="/opt/airflow/dags/data/"+latest_filename.split(".")[0]+".csv"
    trip_data_df.to_csv(latest_filename.split(".")[0]+".csv")

    source_file_path=df_csv_path
    destination_blob_path=f"trip_data/{latest_filename.split('.')[0]}"
    push_to_gcp_bucket(gcp_bucket_name,destination_blob_path,latest_filename.split(".")[0]+".csv")
    return {"df":trip_data_df,"path":"data/"+latest_filename.split(".")[0]+".csv"}
    print(trip_data_df.shape)

def rgx_match(fpath):
  regex_string=r'.+(?:-tripdata.zip)$'
  return re.match(regex_string,fpath)


def push_to_gcp_bucket(bucket_name,destination_blob,source_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/opt/airflow/keys/pedalpulse-440019-919eead68e28.json"

            storage_client=storage.Client()
            bucket=storage_client.get_bucket(bucket_name)
            blob=bucket.blob(destination_blob)
            blob.upload_from_filename(source_path)

            print(f"Dataset {source_path} uploaded to GCP bucket: {bucket_name} at destination {destination_blob}")



if __name__=="__main__":

        get_latest_tripdata_df()