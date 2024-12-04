import logging
from google.cloud import aiplatform, storage

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

# Constants
PROJECT_ID = "blue-bike-prediction"
REGION = "us-central1"
BUCKET_NAME = "test-blue-bikes"
MODEL_FOLDER = "models/"
CONTAINER_URI = "gcr.io/blue-bike-prediction/model-trainer:latest"
MODEL_SERVING_CONTAINER_IMAGE_URI = "gcr.io/blue-bike-prediction/model-serve:latest"
base_output_dir = "gs://test-blue-bikes/models/"
bucket = "gs://test-blue-bikes/models/model/"
ENDPOINT_ID = "blue-bike-endpoint"
# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket)


def create_and_run_training_job(display_name, container_uri, base_output_dir):
    """
    Creates and runs a custom container training job in Vertex AI.

    Args:
        display_name (str): Name of the training job.
        container_uri (str): URI of the container used for training.
        base_output_dir (str): Output directory for the trained model artifacts.

    Returns:
        aiplatform.Model: The trained model artifact.
    """

    logger.info(f"Creating training job: {display_name}")

    # Create the training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=MODEL_SERVING_CONTAINER_IMAGE_URI,
        # staging_bucket=f"gs://{BUCKET_NAME}"
        staging_bucket=bucket,
    )

    logger.info(f"Running training job: {display_name}")

    # Run the training job
    model = job.run(
        model_display_name=display_name,
        base_output_dir=base_output_dir,
        service_account="muskankh03@blue-bike-prediction.iam.gserviceaccount.com",
    )

    logger.info(f"Training job completed. Model saved to: {model.uri}")
    return model


def deploy_model(model):
    try:
        # Check if the endpoint exists
        try:
            endpoint = aiplatform.Endpoint(ENDPOINT_ID)
            endpoint = endpoint.get()  # Try to retrieve the endpoint if it exists
            logger.info(f"Found existing endpoint: {endpoint.resource_name}")
        except Exception as e:
            # If the endpoint is not found, create a new one
            logger.info(f"Endpoint {ENDPOINT_ID} not found. Creating a new endpoint...")
            endpoint = aiplatform.Endpoint.create(
                display_name="blue-bike-endpoint",  # You can choose a different name
                location=REGION,
            )
            logger.info(f"Created new endpoint: {endpoint.resource_name}")

        # Deploy the model to the endpoint
        logger.info(f"Deploying model to endpoint {endpoint.resource_name}...")
        model.deploy(
            deployed_model_display_name="blue-bike-model-deployed",
            endpoint=endpoint,
            machine_type="n1-standard-4",  # You can choose another machine type based on requirements
            sync=True,
        )

        logger.info(f"Model deployed to endpoint: {endpoint.name}")

    except Exception as e:
        logger.error(f"Failed to deploy model: {e}")
        raise


def register_models():
    """Main function to train, discover, and register models dynamically."""
    logger.info(f"Creating and running the training job...")
    trained_model = create_and_run_training_job(
        display_name="Blue-Bike-Prediction",
        container_uri=CONTAINER_URI,
        base_output_dir=f"gs://{BUCKET_NAME}/{MODEL_FOLDER}",
    )

    # Use the parent folder as the artifact_uri
    artifact_uri = f"gs://{BUCKET_NAME}/{MODEL_FOLDER}model/"
    logger.info(f"Registering model from artifact URI: {artifact_uri}")

    return trained_model


if __name__ == "__main__":
    model = register_models()
    deploy_model(model)
