import pandas as pd
import tensorflow_data_validation as tfdv
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to generate statistics and validate schema
def generate_statistics_and_validate_schema(data_path, schema_path):
    try:
        # Load the data
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded from {data_path} successfully.")

        # Generate statistics from the DataFrame
        stats = tfdv.generate_statistics_from_dataframe(df)
        logging.info("Statistics generated successfully.")

        # Write statistics to a file
        tfdv.write_statistics(stats, 'statistics_output')
        logging.info("Statistics written to 'statistics_output'.")

        # Load schema from a file or define inline
        schema = tfdv.load_schema(schema_path)  # Load your schema from the provided schema path
        logging.info(f"Schema loaded from {schema_path} successfully.")

        # Validate the dataset against the schema
        anomalies = tfdv.validate_statistics(stats, schema)

        # Output anomalies if any
        if anomalies:
            logging.warning(f"Data validation issues found: {anomalies}")
        else:
            logging.info("Data validation successful.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example paths for testing
    data_path = 'sample_test_data.csv'  # Path to your sample data CSV
    schema_path = 'schema.pbtxt'  # Path to your schema file

    # Call the function with test data
    generate_statistics_and_validate_schema(data_path, schema_path)
