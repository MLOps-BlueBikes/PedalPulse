# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose the Streamlit port
EXPOSE 8080

# Run Streamlit when the container launches
# CMD ["streamlit", "run", "app.py"]
#CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false", "--browser.serverAddress", "0.0.0.0", "--browser.gatherUsageStats", "false", "--server.port", "8080"]

CMD streamlit run --server.port 8080 --server.enableCORS false app.py