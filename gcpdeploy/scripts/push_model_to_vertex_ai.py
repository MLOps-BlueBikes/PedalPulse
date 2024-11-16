import os
from google.cloud import aiplatform
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def push_model_to_vertex_ai(model_path, project_id, region, model_display_name):
    """Pushes the model from a local path to Vertex AI Model Registry.

    Args:
        model_path (str): Path to the model file.
        project_id (str): Google Cloud Project ID.
        region (str): Google Cloud region.
        model_display_name (str): Display name for the model in Vertex AI.
    """
    # Initialize the Vertex AI client
    aiplatform.init(project=project_id, location=region)

    logger.info(f"Pushing model to Vertex AI Model Registry: {model_path}")

    # Upload model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_path,  # Path to the model folder or file
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest"  # Update for your needs
    )

    logger.info(f"Model pushed successfully. Model resource name: {model.resource_name}")

 
