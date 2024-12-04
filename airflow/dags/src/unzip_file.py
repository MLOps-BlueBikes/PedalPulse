import zipfile
import os
import ast  # For safe evaluation of string lists as lists
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def unzip_file(zip_paths, extract_to="extracted_files"):
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
            logging.info(f"Parsed zip_paths successfully: {zip_paths}")
        except (ValueError, SyntaxError) as e:
            logging.error(f"Error parsing zip_paths: {e}")
            raise e(f"Error parsing zip_paths: {zip_paths}") 

    # Create the extraction directory if it doesn't exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        logging.info(f"Created extraction directory: {extract_to}")

    extracted_files = []
    for zip_path in zip_paths:
        # Check if the path exists
        if not os.path.isfile(zip_path):
            logging.warning(f"File not found: {zip_path}")
            continue

        # Extract files if it exists
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                csv_files = [
                    f
                    for f in z.namelist()
                    if f.endswith(".csv") and not f.startswith("__MACOSX")
                ]
                for csv_file in csv_files:
                    try:
                        z.extract(csv_file, extract_to)
                        extracted_file_path = os.path.join(extract_to, csv_file)
                        extracted_files.append(extracted_file_path)
                        logging.info(f"Extracted {csv_file} to {extracted_file_path}")
                    except EOFError:
                        logging.warning(f"File {csv_file} in {zip_path} is truncated or corrupted and could not be extracted.")
                        raise EOFError(f"File {csv_file} in {zip_path} is truncated or corrupted and could not be extracted.")
        
        except zipfile.BadZipFile as e:
            logging.error(f"File {zip_path} is not a valid ZIP file or is corrupted.")
            raise e(f"File {zip_path} is not a valid ZIP file or is corrupted.")
        except Exception as e:
            logging.error(f"Unexpected error while extracting {zip_path}: {e}")
            raise e(f"Unexpected error while extracting {zip_path}: {e}")

    return extracted_files
