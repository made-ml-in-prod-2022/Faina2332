import os
import pandas as pd
import numpy as np
import click
import pickle


def load_pkl(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


@click.command("predict")
@click.option("--input-dir", "-l", required=True)
@click.option("--model-dir", "-m", required=True)
@click.option("--output-dir", "-s", required=True)
def predict(input_dir: str, model_dir: str, output_dir: str):
    # Loading data, model, transformer
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    model = load_pkl(os.path.join(model_dir, "model.pkl"))
    transformer = load_pkl(os.path.join(model_dir, "transformer.pkl"))
    # Filling missing
    data = data.fillna(data.mean())
    # Transforming data
    X = transformer.transform(data)
    # Predicting
    predictions = model.predict(X)
    predictions = pd.Series(predictions)
    # Saving
    os.makedirs(output_dir, exist_ok=True)
    predictions.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == '__main__':
    predict()
