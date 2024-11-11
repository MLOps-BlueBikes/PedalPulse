import requests
import os
import ast  # for safely evaluating string lists as lists
import logging
from datetime import datetime, timezone
from airflow.exceptions import AirflowSkipException


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def ingest_data(urls, download_dir="downloads", **context):
    """
    Downloads ZIP files from a list of URLs.

    Args:
        urls (list or str): List of URLs to download, or a string representation of a list.
        download_dir (str): Directory to save downloaded files.
    """
    # Ensure urls is a list by evaluating it if it's a string representation
    if isinstance(urls, str):
        try:
            urls = ast.literal_eval(urls)  # Safely evaluate string as list if needed
            logging.info("Parsed URLs successfully.")
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing URLs: {e}")
            raise e 

    logging.info(f"Final list of URLs for ingestion: {urls}")

    # Get the execution date from context
    execution_date = context["execution_date"]
    current_date = datetime.now(timezone.utc)

    # Skip if the execution date is the current month
    if (
        execution_date.year == current_date.year
        and execution_date.month == current_date.month
    ) or (
        execution_date.year == current_date.year
        and execution_date.month == current_date.month - 1
    ):
        logging.info(f"Skipping processing for {execution_date.strftime('%Y-%m')}")
        raise AirflowSkipException

    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        logging.info(f"Created download directory: {download_dir}")

    downloaded_files = []
    for url in urls:
        logging.info(f"Processing URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the request was unsuccessful
            file_path = os.path.join(download_dir, os.path.basename(url))
            with open(file_path, "wb") as f:
                f.write(response.content)
            logging.info(f"Downloaded {file_path}")
            downloaded_files.append(file_path)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")
            raise e 

    return downloaded_files
