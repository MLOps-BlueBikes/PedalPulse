# Import necessary libraries and modules
from airflow import DAG
#from airflow.operators.python_operator import PythonOperator
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
from src.blue_bikes_prediction import load_bike_data, preprocess_data
from src.blue_bikes_prediction import save_to_gcp
#from airflow import configuration as conf
# Enable pickle support for XCom, allowing complex data to be passed between tasks
#conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for the DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=10),  # Delay before retries
}



# Create a DAG instance named 'blue_bike_prediction_dag'
dag = DAG(
    'blue_bike_prediction_dag',
    default_args=default_args,
    description='DAG for Blue Bikes Prediction Project',
    schedule_interval='@daily',  # Change as per the project requirements
    catchup=False,
)

# Define PythonOperators for each function

# Task to load data, calls the 'load_bike_data' Python function
load_data_task = PythonOperator(
    task_id='load_bike_data_task',
    python_callable=load_bike_data,
    dag=dag,
)

# Task to perform data preprocessing, depends on 'load_data_task'
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=preprocess_data,
    #op_args=[load_data_task.output],
    op_args=['{{ ti.xcom_pull(task_ids="load_bike_data_task") }}'], 
    dag=dag,
)

#
# Task to save preprocessed data to GCP
save_to_gcp_task = PythonOperator(
    task_id='save_to_gcp_task',
    python_callable=save_to_gcp,
    #op_args=['your_gcp_bucket_name', 'data/processed_data.csv', 'processed_data.csv'],  # Update these arguments
    op_args=['blue_bikes_bucket', '{{ ti.xcom_pull(task_ids="data_preprocessing_task") }}', 'processed_data.csv'],
  
    dag=dag,
)

# Set task dependencies
load_data_task >> data_preprocessing_task >> save_to_gcp_task



# # Task to build and save a model, depends on 'data_preprocessing_task'
# train_and_save_model_task = PythonOperator(
#     task_id='train_and_save_model_task',
#     python_callable=train_and_save_model,
#     op_args=[data_preprocessing_task.output, "bike_prediction_model.sav"],
#     provide_context=True,
#     dag=dag,
# )

# # Task to evaluate the saved model, depends on 'train_and_save_model_task'
# evaluate_model_task = PythonOperator(
#     task_id='evaluate_model_task',
#     python_callable=evaluate_model,
#     op_args=["bike_prediction_model.sav", train_and_save_model_task.output],
#     dag=dag,
# )


#load_data_task >> data_preprocessing_task >> train_and_save_model_task >> evaluate_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()


# from airflow import DAG
# from airflow.operators.dummy import DummyOperator
# from datetime import datetime

# with DAG('example_dag_start', start_date=datetime(2023, 1, 1), schedule_interval='@daily') as dag:
#     start = DummyOperator(task_id='start')
#     end = DummyOperator(task_id='end')
#     start >> end
    







