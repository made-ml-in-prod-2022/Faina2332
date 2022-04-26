import os
import logging

import pandas as pd
import hydra
from omegaconf import DictConfig

from ml_project.model.fit_predict_model import predict
from ml_project.utils import read_data, load_pkl

from ml_project.entities import PredictPipelineParams, PredictPipelineParamsSchema

logger = logging.getLogger("ml_project/predict_pipeline")


def predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> pd.DataFrame:
    logger.info("Start prediction")
    logger.info("Data loading")
    data = read_data(predict_pipeline_params.input_data_path)

    logger.info("Loading pretrained transformer")
    transformer = load_pkl(predict_pipeline_params.pipeline_path)
    transformed_data = pd.DataFrame(transformer.transform(data))

    logger.info("Loading pretrained ml_project")
    model = load_pkl(predict_pipeline_params.model_path)

    logger.info("Making predictions")
    predictions = predict(model, transformed_data)
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(predict_pipeline_params.output_data_path, header=False)
    logger.info(f"Prediction saved to file{predict_pipeline_params.output_data_path}")

    return predictions


@hydra.main(config_path="../configs", config_name="predict_config")
def start_predict_pipeline(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path(".."))
    schema = PredictPipelineParamsSchema()
    params = schema.load(cfg)
    predict_pipeline(params)


if __name__ == "__main__":
    start_predict_pipeline()
