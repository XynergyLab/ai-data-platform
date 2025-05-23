from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sample_data_processing',
    default_args=default_args,
    description='Sample data processing pipeline',
    schedule_interval=timedelta(days=1),
)

def process_data():
    """Sample data processing function"""
    print("Processing data...")

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)
