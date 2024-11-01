import zipfile
import os
import ast  # For safe evaluation of string lists as lists

def unzip_file(zip_paths, extract_to='extracted_files'):
    """
    Unzips files from a list of ZIP paths.
    
    Args:
        zip_paths (list or str): List of ZIP file paths or a string representation of the list.
        extract_to (str): Directory to save extracted files.
    """
    # Ensure zip_paths is a list by evaluating it if it's a string representation
    if isinstance(zip_paths, str):
        try:
            zip_paths = ast.literal_eval(zip_paths)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing zip_paths: {e}")
            return []

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    extracted_files = []
    for zip_path in zip_paths:
        # Check if the path exists
        if not os.path.isfile(zip_path):
            print(f"File not found: {zip_path}")
            continue
        # Extract files if it exists
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')]
            for csv_file in csv_files:
                try:
                    z.extract(csv_file, extract_to)
                    extracted_files.append(os.path.join(extract_to, csv_file))
                    print(f"Extracted {csv_file}")
                except EOFError:
                    print(f"Warning: File {csv_file} in {zip_path} is truncated or corrupted and could not be extracted.")
    return extracted_files
