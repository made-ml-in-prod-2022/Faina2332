from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

HOST_DIR = r"C:\Users\faina\MADE\2_sem\ML in prod\HW_projects\airflow_ml_dags\data"
DATA_RAW_PATH = "/data/raw/{{ ds }}"


with DAG(
    "download_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(10),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command=f"-s {DATA_RAW_PATH}",
        network_mode="bridge",
        task_id="download",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DIR, target="/data", type='bind')]
    )

    download
