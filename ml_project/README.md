Организация проекта:

    ├── configs            <- Сonfiguration files
    │
    ├── data
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
    ├── ml_project         <- Source code for use in this project.
    │   │
    │   ├── __init__.py    <- Makes a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── entities       <- Scripts to read configuration file
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                    predictions
    │   ├── predict_pipeline.py <- Pipeline for prediction
    │   │
    │   ├── train_pipeline.py <- Pipeline for training model
    │  
    ├── tests                <- Tests folder