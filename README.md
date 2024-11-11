# Pedal Pulse - BlueBikes Demand Prediction

This project aims to predict the demand for BlueBikes using historical data and station information. The goal is to deploy an MLOps pipeline that automates data ingestion, model training, deployment, and retraining. 

## Project Structure
- **data/**: Stores raw and processed datasets.
- **notebooks/**: Contains Jupyter notebooks for data exploration.
- **src/**: The source code for data ingestion, preprocessing, training, and inference.
- **models/**: Stores trained model artifacts.
- **tests/**: Unit and integration tests for various components of the pipeline.
- **docker/**: Docker-related files for containerization.
- **airflow/**: Contains DAGs for automating pipelines using Apache Airflow.
- **ci/**: Files for Continuous Integration/Deployment (CI/CD) setup.
- **requirements.txt**: A list of dependencies required for the project.

## Dataset Information

### **Dataset: BlueBikes Comprehensive Trip Histories & Station Data**

- **Time Period**: Collected from January 2011 to August 2024 (Monthly Drip Data CSVs).
- **Size**: Varies per quarter, around 5 million records per dataset.
  
#### **Data Types**
- **Trip Histories**:
  - **Numerical**: Trip duration, bike ID.
  - **Categorical**: User type, gender.
  - **Time**: Start and stop times.
  - **Geospatial**: Start/end station IDs and names.
  
- **Station Data**:
  - **Numerical**: Station ID, total docks.
  - **Geospatial**: Latitude, longitude.
  - **Categorical**: Municipality, station name.

### **Data Sources**

1. **Trip Histories**: Downloadable files of BlueBikes trip data, updated quarterly from [BlueBikes Trip Data](https://www.bluebikes.com/system-data).
2. **Station Data**: Downloadable station data from [BlueBikes Station Data](https://www.bluebikes.com/system-data).
3. **Real-time Station Data**: Accessible through the General Bikeshare Feed Specification (GBFS) API.

### **Data Card**
<img width="736" alt="Screenshot 2024-11-03 at 3 50 38 PM" src="https://github.com/user-attachments/assets/9c6f43b7-0803-435e-b2ff-945b816cf2bf">


## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- DVC
- Google Cloud SDK (for GCP)
- Apache Airflow (for orchestrating pipelines)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MLOps-BlueBikes/PedalPulse.git
   cd PedalPulse
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Docker Airflow:**
   With Docker running, initialize the database. This step only has to be done once.
   ```docker
   docker compose up airflow-init
   ```
   Run airflow
   ```docker
   docker-compose up
   ```

   Wait until terminal outputs something similar to
   `app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

4. **Check Airflow:**
   
   Visit localhost:8080, login with the following credentials:
   ```
   user: airflow
   password: airflow
   ```
   Run the DAG by clicking on the play button on the right side of the window

5. **Stop Docker containers:**
   ```docker
   docker compose down
   ```

4. **Run Jupyter Notebooks (optional):**
   For data exploration, use Jupyter to run the notebooks in the `notebooks/` folder.

   ```bash
   jupyter notebook
   ```

### Data Pipeline 

### Data Preprocessing

In this project, data preprocessing is a critical step to ensure high-quality input data for our machine learning models, ultimately enhancing the accuracy of demand forecasting for Bluebikes. Below are the key steps involved in preprocessing the data:

1. **Data Collection**  
   We begin by downloading trip data from the official Bluebikes website’s S3 buckets. This data includes information on individual bike trips, such as start and end times, bike type, trip duration, and station details.

2. **Data Type Conversion**  
   To facilitate effective analysis, specific fields are converted to appropriate data types:
   - **Date fields**: Converted to a readable datetime format for temporal analysis.
   - **Categorical fields**: Fields such as membership type and bike type are transformed to categorical types to optimize storage and computation during modeling.

3. **Temporal Feature Extraction**  
   From the trip start and end times, we derive additional temporal features that enhance forecasting accuracy:
   - **Year, month, day, hour**: To capture seasonal, monthly, weekly, and hourly patterns.
   - **Day name**: Useful for distinguishing between weekday and weekend usage.
   - **Trip duration**: Calculated in minutes to assess trip lengths and categorize short vs. long trips.

4. **Handling Missing and Invalid Data**  
   - **Dropping Missing Station IDs**: Rows with missing station IDs are removed to maintain data integrity, as station IDs are crucial for demand forecasting.
   - **Trip Duration Validation**: Trips with a duration less than 5 minutes or exceeding 1440 minutes (24 hours) are excluded.
   - **Trip Distance Validation**: Trips with a distance of less than 0 km are considered invalid and removed.

5. **Data Upload to GCP**  
   After preprocessing, the cleaned dataset is uploaded to Google Cloud Platform (GCP). This allows for scalable data storage and facilitates downstream model training and deployment within the MLOps pipeline.

These preprocessing steps ensure that our data is relevant, consistent, and robust, improving the overall performance and reliability of the demand forecasting model.

### Unit testing

1. **Monthly URL Generation:**
Dynamically generates URLs for monthly Bluebikes data files.

2. **Data Download & Extraction:**
Downloads and extracts data, with fallback to previous months if the file is missing.

3. **Data Quality Tests:**

**Missing Values:** Checks for acceptable levels of missing values in critical columns.
- **Column Data Types:** Validates key column types (e.g., ride_id as string, started_at as datetime).
- **Date Format:** Ensures dates follow YYYY-MM-DD HH:MM:SS.
- **Trip Duration:** Confirms non-negative trip durations.
- **Latitude & Longitude:** Validates coordinates are within range.
- **Unique Ride IDs:** Ensures ride_id values are unique.
- **Membership Type:** Checks member_casual only has member or casual.

### Alerts

Email alerts are configured to notify the owner whenever any task fails. This setup provides proactive monitoring for critical points in the data pipeline, helping maintain seamless data operations.   
- **Ingestion Task Alerts**: Alerts here are essential, as they provide immediate notification if data cannot be fetched from the source (Bluebikes)
- **Preprocessing Task Alerts**: Email alerts during preprocessing allow for swift intervention. This is critical because preprocessing often involves data validation, cleaning, and transformation steps; without real-time alerts, errors could go unnoticed and lead to incorrect final data output.
- **Uploading to Remote GCS Bucket Alerts**: Failure alerts for this task help identify connectivity issues, permissions errors, or storage capacity problems. Immediate notifications, the owner can address these issues without delay, ensuring that data is successfully stored and accessible for future use to prevent data loss. 

### Usage

1. **Data Ingestion:**
   Run the data ingestion pipeline manually or schedule it in Apache Airflow.
   ```bash
   python src/data_pipeline.py
   ```

2. **Train the Model:**
   Train the model using the preprocessed data.
   ```bash
   python src/train_model.py
   ```

3. **Make Predictions:**
   Use the trained model to make predictions.
   ```bash
   python src/predict.py
   ```

4. **Deploy the Model:**
   Containerize the API using Docker and deploy it on GCP or Kubernetes.
   ```bash
   docker build -t bluebikes-api .
   docker run -p 8000:8000 bluebikes-api
   ```

### Model Retraining
The retraining pipeline can be automated using Airflow to trigger the retraining process periodically or when new data becomes available.

### Monitoring and Logging
The system is set up to use **Prometheus** and **Grafana** for monitoring, with **ELK Stack** or **GCP Stackdriver** for logging.
