import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import click


def save_model(model, model_dir):
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


@click.command("train")
@click.option("--input-dir", "-l", required=True)
@click.option("--model-dir", "-m", required=True)
def train(input_dir: str, model_dir: str):
    # Loading data
    X_train = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "train_target.csv"))
    # Training the model
    model = RandomForestClassifier().fit(X_train, y_train)
    # Saving
    os.makedirs(model_dir, exist_ok=True)
    save_model(model, model_dir)


if __name__ == '__main__':
    train()
