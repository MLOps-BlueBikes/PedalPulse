import boto3
from zipfile import ZipFile
import os
import pandas as pd
from pathlib import Path
import re

def get_latest_tripdata_df():

    os.environ['AWS_ACCESS_KEY_ID']= 'XXXXXXXX'
    os.environ['AWS_SECRET_ACCESS_KEY']= 'XXXXXXXX'

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

    l=[]
    i=0
    latest_filename=''
    for obj in bucket.objects.all():
        if re.match(r'.+(?:-tripdata.zip)$',obj._key):
            l.append(obj._key)  # Prints the name of each object in the bucket
            if int(obj._key.split("-")[0])>i:
                i= int(obj._key.split("-")[0])
    for f_name in l:
        if all([str(i) in f_name and '-tripdata.zip' in f_name]):
            latest_filename=f_name 
            break 


    s3.Bucket(bucket_name).download_file(latest_filename, latest_filename)

    with ZipFile(latest_filename,'r') as zip:

        zip.extractall()
        trip_data_df=pd.read_csv(Path(os.getcwd(),latest_filename.split(".")[0]+".csv"))
        os.system(f"rm -rf {latest_filename}")
        lf_csv=latest_filename.split(".")[0]+".csv"
        os.system(f"rm {lf_csv}")
    trip_data_df.dropna(inplace=True)
    path_to_trip_df= f'./data/{lf_csv}'
    trip_data_df.to_csv(path_to_trip_df)
    return path_to_trip_df
