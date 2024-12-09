from google.cloud import monitoring_v3
from google.api import metric_pb2

def create_custom_metric(project_id):
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"
    
    # Define the custom metric descriptor
    descriptor = metric_pb2.MetricDescriptor()
    descriptor.type = "custom.googleapis.com/mse"
    descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type = metric_pb2.MetricDescriptor.ValueType.DOUBLE
    descriptor.description = "Model Mean Squared Error (MSE) percentage"
    
    # Create the custom metric
    request = monitoring_v3.CreateMetricDescriptorRequest(
        name=project_name,
        metric_descriptor=descriptor
    )
    # descriptor = client.create_metric_descriptor(request=request)
    # print(f"Created custom metric: {descriptor.name}")

    try:
        created_descriptor = client.create_metric_descriptor(request=request)
        print(f"Created custom metric: {created_descriptor.name}")
    except Exception as e:
        print(f"Error creating custom metric: {e}")

# Run the function for your project
create_custom_metric("bluebike-443722")
