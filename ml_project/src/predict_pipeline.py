import sys
import logging

import pandas as pd
import click

from ml_project.src.entities import PredictPipelineParams, read_predict_pipeline_params
from ml_project.src.models import predict
from ml_project.src.utils import read_data, load_pkl


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def predict_pipeline(config_path: str) -> pd.DataFrame:

    predict_pipeline_params: PredictPipelineParams = read_predict_pipeline_params(config_path)

    logger.info("Start prediction")
    logger.info("Data loading")
    data = read_data(predict_pipeline_params.input_data_path)

    logger.info("Loading pretrained transformer")
    transformer = load_pkl(predict_pipeline_params.transformer_path)
    transformed_data = pd.DataFrame(transformer.transform(data))

    logger.info("Loading pretrained model")
    model = load_pkl(predict_pipeline_params.model_path)

    logger.info("Making predictions")
    predictions = predict(model, transformed_data)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(predict_pipeline_params.predict_path, header=False)
    logger.info(f"Prediction saved to file{predict_pipeline_params.predict_path}")

    return predictions



@click.command(name='predict_pipeline')
@click.argument('config_path', default='config/predict_config.yaml')
def predict_pipeline_command(config_path: str):
    """ Make start for terminal """
    predict_pipeline(config_path)


if __name__ == '__main__':
    predict_pipeline_command()