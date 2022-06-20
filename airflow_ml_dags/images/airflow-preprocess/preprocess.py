import os
import pandas as pd
import click
import pickle
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir", "-l", required=True)
@click.option("--output-dir", "-s", required=True)
@click.option("--transformer-dir", "-t", required=True)
def preprocess(input_dir: str, output_dir: str, transformer_dir: str):
    # Loading data
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    # A little assert
    assert data.shape[0] == target.shape[0], (
        f"Data_shape: {data.shape[0]}\ny_shape: {target.shape[0]}"
    )
    # Filling missing
    data = data.fillna(data.mean())
    # Fitting a transformer
    transformer = StandardScaler()
    transformer.fit(data)
    data_norm = pd.DataFrame(transformer.transform(data))
    # Saving
    os.makedirs(output_dir, exist_ok=True)
    data_norm.to_csv(os.path.join(output_dir, "preprocessed_data.csv"), index=False)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "transformer.pkl"), "wb") as f:
        pickle.dump(transformer, f)


if __name__ == '__main__':
    preprocess()
