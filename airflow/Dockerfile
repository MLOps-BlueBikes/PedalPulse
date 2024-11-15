# Start from the official Airflow image
FROM  apache/airflow:2.6.3

# Switch to root to install system packages
USER root
# Install Chromium and Chrome driver
RUN apt-get update && apt-get install -y \
    chromium-driver \
    chromium \
    && rm -rf /var/lib/apt/lists/*
# Set environment variables for Chrome
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_DRIVER=/usr/bin/chromedriver

# Switch back to the Airflow user and install Python packages
USER airflow
# Install Python packages
RUN pip install google-cloud-storage 
RUN pip install selenium bs4 lxml selenium-stealth

# Debugging information
RUN python --version
RUN pip --version
RUN airflow version
