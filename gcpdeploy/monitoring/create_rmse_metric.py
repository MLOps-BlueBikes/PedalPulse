from google.cloud import monitoring_v3
from google.protobuf.duration_pb2 import Duration
from google.cloud.monitoring_v3.types import AlertPolicy

def create_alert_policy(project_id):
    # Initialize the Cloud Monitoring client
    client = monitoring_v3.AlertPolicyServiceClient()
    
    # Define the alert policy
    alert_policy = monitoring_v3.AlertPolicy(
        display_name="Model RMSE metric Alert",
        notification_channels=["projects/bluebike-443722/notificationChannels/10947410200877297806"],
        conditions=[],
        #alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(),
        combiner=AlertPolicy.ConditionCombinerType.OR  # or use AND or AND_WITH_MATCHING_RESOURCE
    )
    
    # Specify the metric to monitor (RMSE metric from Cloud Monitoring)
    condition = monitoring_v3.AlertPolicy.Condition(
        display_name="RMSE below threshold",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter='resource.type="global" AND metric.type="custom.googleapis.com/rmse"',
            comparison=monitoring_v3.ComparisonType.COMPARISON_LT,  # Less than threshold
            threshold_value=0.5,  # Set the threshold for RMSE
            duration=Duration(seconds=60),  # Condition must be true for this long to trigger
            aggregations=[
                monitoring_v3.Aggregation(
                    alignment_period=Duration(seconds=60),  # Adjust the evaluation period
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                )
            ]
        )
    )
    alert_policy.conditions.append(condition)
    
    # Create the alert policy
    request = monitoring_v3.CreateAlertPolicyRequest(
        name=f"projects/{project_id}",
        alert_policy=alert_policy
    )
    created_alert_policy = client.create_alert_policy(request=request)
    print(f"Alert policy created: {created_alert_policy.name}")

# Usage
create_alert_policy("bluebike-443722")
