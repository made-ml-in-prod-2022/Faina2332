import os
import logging
from typing import Dict

import pandas as pd
import hydra
from omegaconf import DictConfig

from ml_project.data import split_train_val_data
from ml_project.features import extract_target, column_transformer
from ml_project.model import train, predict, evaluate
from ml_project.entities import TrainPipelineParams, TrainPipelineParamsSchema
from ml_project.utils import read_data, save_metrics_json, save_pkl

logger = logging.getLogger("ml_project/train_pipeline")


def train_pipeline(train_pipeline_params: TrainPipelineParams) -> Dict[str, float]:
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

    logger.info("Training ml_project")
    model = train(train_features, train_target, train_pipeline_params.train_params)

    logger.info("Evaluating ml_project")
    val_features = pd.DataFrame(transformer.transform(val_data))
    val_target = extract_target(val_data, train_pipeline_params.feature_params)
    predictions = predict(model, val_features)
    metrics = evaluate(predictions, val_target)
    logger.info(f"Model scores: {metrics}")
    logger.info("Saving ml_project and metrics")
    save_pkl(model, train_pipeline_params.output_model_path)
    save_metrics_json(metrics, train_pipeline_params.metrics_path)

    return metrics


@hydra.main(config_path="../configs", config_name="train_config")
def start_train_pipeline(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainPipelineParamsSchema()
    params = schema.load(cfg)
    train_pipeline(params)


if __name__ == "__main__":
    start_train_pipeline()
