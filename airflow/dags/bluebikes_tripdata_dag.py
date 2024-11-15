from src.s3_connect import connect_to_bucket, get_trip_dataset
from src.unzip_and_extract import extract_zip_to_csv
from src.gcp_ops import push_to_gcp_bucket,pull_from_gcp_bucket
from src.preprocess_ride_data import clean_and_preprocess_data
from src.weather_data import process_bike_trip_weather_data
from airflow.operators.python import PythonOperator
from airflow import DAG
import pendulum



# Define the DAG
with DAG(
    dag_id="monthly_tripdata_dag",
    schedule_interval="0 0 7 * *",
    start_date=pendulum.datetime(2024, 3, 7, tz="UTC"),
    description="DAG to process Bluebikes trip data",
    catchup=False,
    default_args={'email': ["mlopsgcpproject@gmail.com"],
                  'email_on_failure': True,
                  'email_on_retry': False}
) as dag:



    connect_to_bucket_task = PythonOperator(
        task_id='connect_to_bucket',
        python_callable=connect_to_bucket
    )



    get_trip_dataset_task = PythonOperator(
        task_id='get_trip_dataset',
        python_callable=get_trip_dataset,
        op_kwargs={'bucket_name': '{{ task_instance.xcom_pull(task_ids="connect_to_bucket") }}', 'fetch_date': '{{ds}}'},
        provide_context=True
    )

    
    extract_zip_to_csv_task = PythonOperator(
        task_id='extract_zip_to_csv',
        python_callable=extract_zip_to_csv,
        op_kwargs={'latest_filename': '{{ task_instance.xcom_pull(task_ids="get_trip_dataset") }}'}
    )

 
    push_to_gcp_bucket_task = PythonOperator(
        task_id='push_to_gcp_bucket',
        python_callable=push_to_gcp_bucket,
        op_kwargs={
            'destination_blob_path': '{{ task_instance.xcom_pull(task_ids="extract_zip_to_csv", key="return_value")["blob_path"] }}',
            'source_path': '{{ task_instance.xcom_pull(task_ids="extract_zip_to_csv", key="return_value")["df_path"] }}'
        },
        provide_context=True
    )

    pull_from_gcp_bucket_task= PythonOperator(
                task_id='pull_from_gcp_bucket',
                python_callable=pull_from_gcp_bucket,
                op_kwargs={
                    'destination_blob_path': '{{ task_instance.xcom_pull(task_ids="push_to_gcp_bucket", key="return_value") }}',
                },
                provide_context=True
            )

    preprocess_task = PythonOperator(
        task_id = 'preprocess_tripdata',
        python_callable=clean_and_preprocess_data,
        op_kwargs={'filepath':'{{ task_instance.xcom_pull(task_ids="pull_from_gcp_bucket", key="return_value") }}'}
    )
    
    '''

    weather_integration_task = PythonOperator(
        task_id = 'weather_integration',
        python_callable=process_bike_trip_weather_data,
        op_kwargs={'preprocessed_filepath':'{{ task_instance.xcom_pull(task_ids="preprocess_task", key="return_value") }}'}
    )
    '''

    # Set task dependencies
    connect_to_bucket_task >> get_trip_dataset_task >> extract_zip_to_csv_task >> push_to_gcp_bucket_task >> pull_from_gcp_bucket_task>>preprocess_task






