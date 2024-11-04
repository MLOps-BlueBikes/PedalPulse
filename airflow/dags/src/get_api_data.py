import pandas as pd
import requests
from datetime import datetime
from google.cloud import storage
def load_data():
    '''Load the data.'''
    station_status_url = "https://gbfs.lyft.com/gbfs/1.1/bos/en/station_status.json"
    station_info_url = "https://gbfs.lyft.com/gbfs/1.1/bos/en/station_information.json"
    output_paths = []
    try:
        status_response = requests.get(station_status_url)
        # Check if the request was successful
        if status_response.status_code == 200:
            status_data = status_response.json()
            station_status = status_data['data']['stations']
            station_status_df = pd.DataFrame(station_status)
            status_output_path = '/opt/airflow/dags/data/station_status.csv'
            station_status_df.to_csv(status_output_path, index=False)
            output_paths.append(status_output_path)
        else:
            print(f"Failed to retrieve data. Status code: {status_response.status_code}")
        info_response = requests.get(station_info_url)
        # Check if the request was successful
        if info_response.status_code == 200:
            info_data = info_response.json()
            station_info = info_data['data']['stations']
            station_info_df = pd.DataFrame(station_info)
            info_output_path = '/opt/airflow/dags/data/station_info.csv'
            station_info_df.to_csv(info_output_path, index=False)
            output_paths.append(info_output_path)
        else:
            print(f"Failed to retrieve data. Status code: {info_response.status_code}")
        output_paths = [status_output_path, info_output_path]
        print(f'Data saved at {output_paths[0]} AND {output_paths[1]}')
        return output_paths
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def preprocess_data(output_paths):
    '''Preprocessing the files to create a structured dataset.'''
    try:
        status_file_path, info_file_path = output_paths
        station_status_df = pd.read_csv(status_file_path)
        station_info_df = pd.read_csv(info_file_path)
        merged_df = pd.merge(station_info_df, station_status_df, on='station_id', how='inner')
        cleaned_df = merged_df[['station_id', 'short_name', 'name', 'num_bikes_available', 'num_ebikes_available', 'num_docks_available', 'capacity', 'last_reported']]
        cleaned_df['num_cl_bikes_available'] = cleaned_df['num_bikes_available'] - cleaned_df['num_ebikes_available']
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
        output_path = f'/opt/airflow/dags/data/data_{formatted_time}.csv'
        cleaned_df.to_csv(output_path, index=False)
        print(f'Data preprocessing completed. Data saved at: {output_path}')
        return output_path
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None
    
def save_to_gcp(bucket_name, source_file, destination_blob):
    '''Save the data generated to GCP.'''
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(source_file)
        print(f"File {source_file} uploaded to {destination_blob} in bucket {bucket_name}.")
    except Exception as e:
        print(f'Error saving data: {e}')
        return None