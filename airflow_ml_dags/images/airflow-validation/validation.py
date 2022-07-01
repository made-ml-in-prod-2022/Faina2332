import os

import pandas as pd
import numpy as np

import pickle
import json
import click

from sklearn.metrics import accuracy_score, f1_score


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


@click.command("validation")
@click.option("--input-dir", "-l", required=True)
@click.option("--model-dir", "-m",required=True)
def validation(input_dir: str, model_dir: str):
    # Loading data, model
    val_data = pd.read_csv(os.path.join(input_dir, "val_data.csv"))
    val_target = pd.read_csv(os.path.join(input_dir, "val_target.csv"))
    model = load_model(os.path.join(model_dir, "model.pkl"))
    # Predicting
    val_predict = model.predict(val_data)
    accuracy = accuracy_score(val_target, val_predict)
    f1 = f1_score(val_target, val_predict)

    metrics = {"accuracy": accuracy, "f1-score": f1}
    # Saving
    with open(os.path.join(model_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == '__main__':
    validation()
