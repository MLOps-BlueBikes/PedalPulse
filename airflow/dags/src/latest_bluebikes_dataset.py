import boto3
from zipfile import ZipFile
import os
import pandas as pd
from pathlib import Path
import re
from google.cloud import storage
from io import BytesIO


def get_latest_tripdata_df():

    os.environ['AWS_ACCESS_KEY_ID']= 'XXXXXXX'
    os.environ['AWS_SECRET_ACCESS_KEY']= 'XXXXXXX'

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

    # Specify the bucket name
    bucket_name = 'hubway-data'

    temp_l=list(filter(rgx_match,[obj._key for obj in bucket.objects.all()]))

    latest_filename=sorted(temp_l,key= lambda s:int(s[:6]),reverse=True)[0]


    s3.Bucket(bucket_name).download_file(Key=latest_filename,Filename="./dummy.zip")

    with ZipFile("./dummy.zip",'r') as zip:
        bytes_data=zip.read(latest_filename.split(".")[0]+".csv")
        trip_data_df = pd.read_csv(BytesIO(bytes_data))

    trip_data_df.dropna(inplace=True)
    df_csv_path="data/"+latest_filename.split(".")[0]+".csv"
    trip_data_df.to_csv(df_csv_path)
    source_file_path=df_csv_path
    destination_blob_path=f"trip_data/{latest_filename.split('.')[0]}"
    push_to_gcp_bucket(gcp_bucket_name,destination_blob_path,source_file_path)
    return {"df":trip_data_df,"path":"data/"+latest_filename.split(".")[0]+".csv"}
    print(trip_data_df.shape)

def rgx_match(fpath):
  regex_string=r'.+(?:-tripdata.zip)$'
  return re.match(regex_string,fpath)


def push_to_gcp_bucket(bucket_name,destination_blob,source_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/path/to/pedalpulse-440019-919eead68e28.json"

            storage_client=storage.Client()
            bucket=storage_client.get_bucket(bucket_name)
            blob=bucket.blob(destination_blob)
            blob.upload_from_filename(source_path)

            print(f"Dataset {source_path} uploaded to GCP bucket: {bucket_name} at destination {destination_blob}")



if __name__=="__main__":

        get_latest_tripdata_df()