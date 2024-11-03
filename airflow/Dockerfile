# Start from the official Airflow image
FROM  apache/airflow:2.6.3
# Set environment variables
#ENV AIRFLOW_HOME=/opt/airflow

# Install Google Cloud SDK and any other packages apache/airflow:2.6.3
RUN pip install google-cloud-storage 
#RUN pip install 'dvc[gcs]' fsspec==2023.1.0
RUN pip install 'dvc[gcs]==2.7.3' fsspec==2022.1.0


# Debugging information
RUN python --version
RUN pip --version
RUN airflow version
RUN dvc --version 