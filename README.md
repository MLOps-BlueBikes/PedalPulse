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

## Getting Started

### Prerequisites
- Python 3.8+
- Docker
- Google Cloud SDK (for GCP)
- Apache Airflow (for orchestrating pipelines)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MuskanKhandelwal/Bluebikes-demand-forecasting.git
   cd Bluebikes-demand-forecasting
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Airflow:**
   Follow instructions to install and configure Apache Airflow.

4. **Run Jupyter Notebooks (optional):**
   For data exploration, use Jupyter to run the notebooks in the `notebooks/` folder.

   ```bash
   jupyter notebook
   ```

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

