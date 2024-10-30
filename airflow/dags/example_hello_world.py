from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime

with DAG(
    dag_id='hello_world_dag',
    start_date=datetime.datetime(2023, 10, 5),
    tags=['example', 'python'],
) as dag:
    task_1 = BashOperator(
        task_id='hello_world_task1',
        bash_command="echo 'Hello World!'"
    )
    task_2 = BashOperator(
        task_id='hello_world_task2',
        bash_command="echo 'Hello World! Now I am triggered after the first task!'"
    )

    task_1 >> task_2
