from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
import time
from google.oauth2 import service_account

def report_model_metric(project_id, accuracy_value, metric):

    # Create credentials using the service account key file
    credentials = service_account.Credentials.from_service_account_file('key.json', scopes=["https://www.googleapis.com/auth/monitoring"])

    client = monitoring_v3.MetricServiceClient(credentials=credentials)
    project_name = f"projects/{project_id}"
    
    # Set up the time series data
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric}"
    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id
    
    # Create a new point
    point = monitoring_v3.Point()
    point.value.double_value = accuracy_value
    
    # Set the time for the point
    now = time.time()
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromSeconds(int(now))
    interval = monitoring_v3.TimeInterval({"end_time": timestamp})
    point.interval = interval
    
    # Add the point to the series
    series.points = [point]
    
    # Write the time series data
    request = monitoring_v3.CreateTimeSeriesRequest(
        name=project_name,
        time_series=[series]
    )
    client.create_time_series(request=request)
    print(f"Reported {metric}: {accuracy_value}")
