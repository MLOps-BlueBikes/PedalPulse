import zipfile
import os

def unzip_file(zip_paths, extract_to='extracted_files'):
    """
    Unzips files from a list of ZIP paths.
    
    Args:
        zip_paths (list): List of ZIP file paths.
        extract_to (str): Directory to save extracted files.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    extracted_files = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Filter out unwanted files like '__MACOSX'
            csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')]
            for csv_file in csv_files:
                try:
                    z.extract(csv_file, extract_to)
                    extracted_files.append(os.path.join(extract_to, csv_file))
                    print(f"Extracted {csv_file}")
                except EOFError:
                    print(f"Warning: File {csv_file} in {zip_path} is truncated or corrupted and could not be extracted.")
                    continue
    return extracted_files



