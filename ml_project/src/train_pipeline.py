import logging
import sys
from typing import Dict

import pandas as pd
import click


from data import split_train_val_data
from features import extract_target, column_transformer
from models import train, predict, evaluate
from entities import TrainPipelineParams, read_train_pipeline_params
from utils import read_data, save_metrics_json, save_pkl

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def train_pipeline(config_path: str) -> Dict[str, float]:

    train_pipeline_params: TrainPipelineParams = read_train_pipeline_params(config_path)
    logger.info(f"Train pipeline parameters {train_pipeline_params}")
    logger.info(f"Model type: {train_pipeline_params.train_params.model_type}")

    logger.info("Data loading")
    data = read_data(train_pipeline_params.input_data_path)
    train_data, val_data = split_train_val_data(data, train_pipeline_params.splitting_params)

    logger.info("Preprocessing data")
    transformer = column_transformer(train_pipeline_params.feature_params)
    transformer.fit(train_data)
    save_pkl(transformer, train_pipeline_params.output_transformer_path)

    train_features = pd.DataFrame(transformer.transform(train_data))
    train_target = extract_target(train_data, train_pipeline_params.feature_params)

    logger.info("Training model")
    model = train(train_features, train_target, train_pipeline_params.train_params)

    logger.info("Evaluating model")
    val_features = pd.DataFrame(transformer.transform(val_data))
    val_target = extract_target(val_data, train_pipeline_params.feature_params)
    predictions = predict(model, val_features)
    metrics = evaluate(predictions, val_target)
    logger.info(f"Model scores: {metrics}")
    logger.info("Saving model and metrics")
    save_pkl(model, train_pipeline_params.output_model_path)
    save_metrics_json(train_pipeline_params.metrics_path, metrics)

    return metrics


@click.command(name='train_pipeline')
@click.argument('config_path', default='config/train_config.yaml')
def train_pipeline_command(config_path: str):
    """ Make start for terminal """
    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()
