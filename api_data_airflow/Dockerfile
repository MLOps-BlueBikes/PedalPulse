# Start from the official Airflow image
#FROM apache/airflow:2.6.3
FROM  apache/airflow:2.6.3
# Set environment variables
# ENV AIRFLOW_HOME=/opt/airflow

# Install Google Cloud SDK and any other packages apache/airflow:2.6.3
RUN pip install google-cloud-storage 
# If you have other dependencies, list them here
# RUN pip install <other-dependency>
RUN pip install pandas
RUN pip install requests

# Debugging information
RUN python --version
RUN pip --version
RUN airflow version