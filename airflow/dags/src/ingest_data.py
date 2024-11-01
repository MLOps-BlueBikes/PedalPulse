import requests
import os
import ast  # for safely evaluating string lists as lists
from datetime import datetime, timezone

def ingest_data(urls, download_dir='downloads', **context):
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
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing urls: {e}")
            return []

    print("Final list of URLs for ingestion:", urls)

    # Get the execution date from context
    execution_date = context['execution_date']
    current_date = datetime.now(timezone.utc)

    # Skip if the execution date is the current month or last month
    if (execution_date.year == current_date.year and execution_date.month == current_date.month) or \
       (execution_date.year == current_date.year and execution_date.month == current_date.month - 1):
        print(f"Skipping processing for {execution_date.strftime('%Y-%m')}")
        return []

    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    downloaded_files = []
    for url in urls:
        print("Processing URL:", url)
        try:
            response = requests.get(url)
            file_path = os.path.join(download_dir, os.path.basename(url))
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_path}")
            downloaded_files.append(file_path)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

    return downloaded_files
