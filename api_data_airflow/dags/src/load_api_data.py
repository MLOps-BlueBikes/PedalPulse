import pandas as pd
import requests

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_api_data():
    """Load the data."""

    station_status_url = "https://gbfs.lyft.com/gbfs/1.1/bos/en/station_status.json"
    station_info_url = "https://gbfs.lyft.com/gbfs/1.1/bos/en/station_information.json"
    output_paths = []

    try:

        status_response = requests.get(station_status_url)
        # Check if the request was successful
        if status_response.status_code == 200:
            status_data = status_response.json()
            station_status = status_data["data"]["stations"]
            station_status_df = pd.DataFrame(station_status)
            status_output_path = "/opt/airflow/dags/data/station_status.csv"
            station_status_df.to_csv(status_output_path, index=False)
            output_paths.append(status_output_path)
            logging.info(
                f"Succesfully downloaded station status data at: {status_output_path}"
            )

        else:
            logging.error(
                f"Failed to retrieve data. Status code: {status_response.status_code}"
            )

        info_response = requests.get(station_info_url)
        # Check if the request was successful
        if info_response.status_code == 200:
            info_data = info_response.json()
            station_info = info_data["data"]["stations"]
            station_info_df = pd.DataFrame(station_info)
            info_output_path = "/opt/airflow/dags/data/station_info.csv"
            station_info_df.to_csv(info_output_path, index=False)
            output_paths.append(info_output_path)
            logging.info(
                f"Succesfully downloaded station info data at: {info_output_path}"
            )

        else:
            logging.error(
                f"Failed to retrieve data. Status code: {info_response.status_code}"
            )

        output_paths = [status_output_path, info_output_path]
        logging.info(f"Data saved at {output_paths[0]} AND {output_paths[1]}")
        return output_paths

    except Exception as e:
        logging.exception(f"Error loading data: {e}")
        return None
