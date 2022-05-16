# HW 1 Ml_project

Dataset : https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci 

Installation:

python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

Usage:

python ml_example/train_pipeline.py configs/train_config.yaml
python ml_example/predict_pipeline.py configs/predict_config.yaml

Test:

pytest tests/

Project Organization
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── configs            <- Сonfiguration files
    │
    ├── data
    │   ├── predict        <- predictions .csv files
    │   │
    │   └── raw            <- original .csv files
    │
    ├── models             <- Trained model, transformers and metrics
    │
    ├── notebooks          <- Jupyter notebooks - EDA
    │
    ├── requirements.txt   <- Requirements
    │
    ├── setup.py           <- Makes project pip installable
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── __init__.py    <- Makes a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │
    │   ├── entities       <- Scripts to read configuration file
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                    predictions
    │   ├── utils          <- Scripts to load, save and read
    │   │  
    │   ├── predict_pipeline.py <- Pipeline for prediction
    │   │
    │   └── train_pipeline.py <- Pipeline for training model
    │  
    └── tests                <- Tests folder