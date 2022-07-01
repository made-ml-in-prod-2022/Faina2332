import os

import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command("split")
@click.option("--input-dir", "-l", required=True)
@click.option("--output-dir", "-s", required=True)
@click.option("--train-size", "-ts", default=0.8)
@click.option("--random-state", "-rs", default=42)
def split(input_dir: str, output_dir: str, train_size: float, random_state: int):
    # Loading data
    data = pd.read_csv(os.path.join(input_dir, "preprocessed_data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    # Splitting the data
    train_data, val_data, train_target, val_target = train_test_split(
        data,
        target,
        random_state=random_state,
        train_size=train_size
    )
    # Saving
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    train_target.to_csv(os.path.join(output_dir, "train_target.csv"), index=False)
    val_target.to_csv(os.path.join(output_dir, "val_target.csv"), index=False)


if __name__ == '__main__':
    split()
