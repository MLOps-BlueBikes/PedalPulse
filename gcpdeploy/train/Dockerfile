# Use a lightweight Python image as base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (scripts, data)
COPY . .

# Create directory for MLflow logs
RUN mkdir -p /app/mlruns
# Set PYTHONPATH to ensure imports work
ENV PYTHONPATH=/app

# Add this to your Dockerfile to suppress Git-related warnings
ENV GIT_PYTHON_REFRESH=quiet

# Set the command to run the training script
CMD ["python", "scripts/model.py"]
