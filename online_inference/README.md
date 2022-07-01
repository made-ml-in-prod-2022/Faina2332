# HW 2 Online_inference
Installation:

python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt

Project Organization

    ├── models                    <- Trained and serialized models  
    │     └── model.pkl           <- Pre-trained model  
    │     └── transformer.pkl     <- Pre-trained transformer  
    ├── Dockerfile                <- Dockerfile to create docker image  
    ├── app.py                    <- FastAPI application  
    ├── make_request.py           <- Script to generate requests for API  
    ├── README.md                 <- The top-level README for developers using this project  
    ├── requiriments.txt          <- The requirements file  
    └── test_app.py               <- Test for API  

Usage for the project

Build Docker image:
~~~
docker build -t fuzzy2332/online_inference:v1 .
~~~

Pull Docker image from Docker hub  :
~~~
docker pull fuzzy2332/online_inference:v1
~~~

Use container:
~~~
docker run -p 8000:8000 fuzzy2332/online_inference:v1
~~~

Make requests:
~~~
python make_request.py
~~~

Tests:
~~~
pytest
~~~

Оптимизация размера docker image:

* Выбрана легковесная версия базового образа (python:slim);
* Установка пакетов через pip без cache (--no-cache-dir);

В итоге образ занимает 548.68 MB
