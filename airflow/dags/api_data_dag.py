from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
from src.load_api_data import load_api_data
from src.preprocess_api_data import preprocess_api_data
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
#conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'ritika dhall',
    'start_date': datetime(2023, 10, 28),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

dag = DAG(
    'get_api_dag',
    default_args=default_args,
    description='DAG to get API data',
    schedule='0 * * * *',
    catchup=False,
)

# Task to load data, calls the 'load_data' Python function
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_api_data,
    dag=dag,
)

# Task to perform data preprocessing, depends on 'load_data_task'
preprocess_data_task = PythonOperator(
    task_id='preprocess_data_task',
    python_callable=preprocess_api_data,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task to save preprocessed file to GCP
# save_to_gcp_task = PythonOperator(
#     task_id='save_to_gcp_task',
#     python_callable=save_to_gcp,
#     op_kwargs={
#         'bucket_name': 'your-gcs-bucket-name',
#         'source_file_name': preprocess_data_task.output,
#         'destination_blob_name': 'processed_data.csv'
#     },
#     dag=dag
# )

# Set task dependencies
load_data_task >> preprocess_data_task

if __name__ == "__main__":
    dag.cli()