import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(
    filename='ingest_data.log',  # Log to a file
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    filemode='a'  # Append to the log file
)

LOCAL_DATA_DIR = 'dags/data'

def load_blue_bikes_data():
    global LOCAL_DATA_DIR
    file_name = '202401-bluebikes-tripdata.csv'
    file_path = os.path.join(LOCAL_DATA_DIR, file_name)
    
    if os.path.exists(file_path):
        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Bluebikes data successfully loaded. Shape of the data: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Failed to load Bluebikes data: {e}")
            return None
    else:
        logging.warning(f"File not found: {file_path}")
        return None

def load_weather_data():
    global LOCAL_DATA_DIR
    file_name = 'boston_climate_data_Jan24.csv'
    file_path = os.path.join(LOCAL_DATA_DIR, file_name)
    
    if os.path.exists(file_path):
        # Read the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(file_path)
            print(f"Weather data successfully loaded. Shape of the data:  {df.shape}")
            logging.info(f"Weather data successfully loaded. Shape of the data: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Failed to load weather data: {e}")
            return None
    else:
        logging.warning(f"File not found: {file_path}")
        return None
