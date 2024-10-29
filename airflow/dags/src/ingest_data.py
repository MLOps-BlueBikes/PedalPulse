


# ingest_data.py
import requests
import os

def ingest_data(urls,download_dir='downloads'):
    """
    Downloads ZIP files from a list of URLs.
    
    Args:
        urls (list): List of URLs to download.
        download_dir (str): Directory to save downloaded files.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for url in urls:
        response = requests.get(url)
        file_path = os.path.join(download_dir, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_path}")
    
    return [os.path.join(download_dir, os.path.basename(url)) for url in urls]

if __name__ == "__main__":
    urls = [
        'https://s3.amazonaws.com/hubway-data/202310-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202311-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202312-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202401-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202402-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202403-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202404-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202405-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202406-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202407-bluebikes-tripdata.zip',
        'https://s3.amazonaws.com/hubway-data/202408-bluebikes-tripdata.zip', 
        'https://s3.amazonaws.com/hubway-data/202409-bluebikes-tripdata.zip'
        
    ]
    path= ingest_data(urls)
