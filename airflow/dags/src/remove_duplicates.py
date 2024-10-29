import pandas as pd

def remove_duplicates(file_path, output_file='final_data.csv'):
    """
    Loads the entire combined CSV, removes duplicates, and writes to a new file.
    
    Args:
        file_path (str): Path to the combined CSV file with possible duplicates.
        output_file (str): File to save the deduplicated data.
    """
    df = pd.read_csv(file_path, dtype=str)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Duplicates removed and data saved to {output_file}")
