'''import pandas as pd
import numpy as np
from google.cloud import storage

# Function to load bike data
def load_bike_data():
    """Loads the Blue Bikes dataset."""
    # try:
    #     data = pd.read_csv('/opt/airflow/dags/data/201501-hubway-tripdata_2.csv')  # Update the path as needed
    #     print("Data loaded successfully.")
    #     return data
    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     return None
    try:
        data = pd.read_csv('/opt/airflow/dags/data/201501-hubway-tripdata_2.csv')  # Path to the raw data file
        print("Data loaded successfully.")
        output_path = '/opt/airflow/dags/data/raw_bike_data.csv'
        data.to_csv(output_path, index=False)  # Save the loaded data
        return output_path  # Return the path to the saved file
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def preprocess_data(file_path):
    """Preprocesses the Blue Bikes dataset."""
    try:
        # Load data from the file path provided by load_bike_data
        data = pd.read_csv(file_path)
        
        # Preprocess the data
        data = data.dropna()  # Remove rows with missing values
        
        output_path = '/opt/airflow/dags/data/processed_data.csv'
        data.to_csv(output_path, index=False)
        print(f"Data preprocessing completed. Processed data saved at: {output_path}")
        return output_path  # Return the path of the preprocessed file
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return None

# # Function to preprocess data
# def preprocess_data(data):
#     """Preprocesses the Blue Bikes dataset by handling missing values."""
#     try:
#         data = data.dropna()
#         output_path = '/opt/airflow/dags/data/processed_data.csv'
#         data.to_csv(output_path, index=False)
#         print("Data preprocessing completed.")
#         return output_path
#     except Exception as e:
#         print(f"Error in preprocessing data: {e}")
#         return None
def save_to_gcp(bucket_name, source_file, destination_blob):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    blob.upload_from_filename(source_file)
    print(f"File {source_file} uploaded to {destination_blob}.")


# def test_gcp_connection():
#     """Function to test GCP connection and storage"""
#     try:
#         client = storage.Client()
#         buckets = list(client.list_buckets())
#         print("Buckets in GCP:", buckets)
#     except Exception as e:
#         print("Error connecting to GCP:", e)


# # Function to train and save the model
# def train_and_save_model(X, y, model_path="bike_prediction_model.sav"):
#     """
#     Trains a RandomForest model on the processed data and saves the model.
#     """
#     try:
#         # Split the data into training and validation sets
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train a RandomForestRegressor model
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
#         model.fit(X_train, y_train)

#         # Save the model
#         with open(model_path, 'wb') as file:
#             pickle.dump(model, file)

#         print("Model trained and saved successfully.")
#         return model_path
#     except Exception as e:
#         print(f"Error in model training: {e}")
#         return None

# # Function to load and evaluate the model
# def evaluate_model(model_path, X, y):
#     """
#     Loads a trained model, makes predictions, and evaluates its performance.
#     """
#     try:
#         # Load the model
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)

#         # Make predictions
#         predictions = model.predict(X)

#         # Evaluate the model
#         rmse = np.sqrt(mean_squared_error(y, predictions))
#         print(f"Model evaluation completed. RMSE: {rmse}")

#         return rmse
#     except Exception as e:
#         print(f"Error in model evaluation: {e}")
#         return None


'''
