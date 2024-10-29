# load_data.py
import pandas as pd

def load_data(csv_files, output_file='combined_data.csv', chunksize=10000):
    """
    Loads CSV files in chunks, handles missing values, and appends to a single output CSV.
    
    Args:
        csv_files (list): List of CSV file paths.
        output_file (str): File to save the combined data.
        chunksize (int): Number of rows per chunk.
    """
    first_chunk = True  # Control for writing header only once
    for file in csv_files:
        for chunk in pd.read_csv(file, chunksize=chunksize, dtype=str):
            # Handle missing values in each chunk
            chunk.fillna(method='ffill', inplace=True)  # Example: forward fill
            chunk.to_csv(output_file, index=False, header=first_chunk, mode='a')
            first_chunk = False
    print(f"Data combined and written to {output_file} with missing values handled.")
