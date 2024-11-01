from google.cloud import storage
import time
import os
import ast  
def upload_to_gcs(bucket_name, file_paths):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    # Ensure urls is a list by evaluating it if it's a string representation
    if isinstance(file_paths, str):
        try:
            file_paths = ast.literal_eval(file_paths)  # Safely evaluate string as list if needed
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing urls: {e}")
            return []
    for file_path in file_paths:
        blob_name = os.path.basename(file_path)
        blob = bucket.blob(blob_name)
        
        retry_count = 0
        max_retries = 8
        while retry_count < max_retries:
            try:
                blob.upload_from_filename(file_path)
                print(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
                break
            except Exception as e:
                print(f"Error uploading {file_path}: {e}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff
                if retry_count == max_retries:
                    print(f"Failed to upload {file_path} after {max_retries} retries")
