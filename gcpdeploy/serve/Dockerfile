FROM python:3.9-slim

# Installs additional packages
#RUN pip install flask scikit-learn==0.24.2 google-cloud-storage pandas

# Install core dependencies
RUN pip install \
    pandas \
    scikit-learn==0.24.2 \
    joblib \
    flask \
    gunicorn \
    google-cloud-storage \
    gcsfs \
    numpy==1.26.4

ENV AIP_STORAGE_URI=gs://test-blue-bikes/models/model/
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
# Copies the API code to the docker image.
COPY . ./

# Expose the port Flask will run on
EXPOSE 8080

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "serving_model.py"]