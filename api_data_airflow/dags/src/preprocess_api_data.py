import pandas as pd
from datetime import datetime

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_api_data(output_paths):
    '''Preprocessing the files to create a structured dataset.'''
    try:

        status_file_path, info_file_path = output_paths
        station_status_df = pd.read_csv(status_file_path)
        station_info_df = pd.read_csv(info_file_path)
        merged_df = pd.merge(station_info_df, station_status_df, on='station_id', how='inner')

        cleaned_df = merged_df[['station_id', 'short_name', 'name', 'num_bikes_available', 'num_ebikes_available', 'num_docks_available', 'capacity', 'last_reported']]
        cleaned_df['num_cl_bikes_available'] = cleaned_df['num_bikes_available'] - cleaned_df['num_ebikes_available']

        output_path = f'/opt/airflow/dags/data/api_data_preprocessed.csv'
        cleaned_df.to_csv(output_path, index=False)
        logging.info(f'Data preprocessing completed. Data saved at: {output_path}')
        return output_path
    
    except Exception as e:
        logging.exception(f"Error preprocessing data: {e}")
        return None