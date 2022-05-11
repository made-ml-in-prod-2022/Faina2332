from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class PredictPipelineParams:
    """ Defines src and data predict pipeline path. """
    input_data_path: str
    predict_path: str
    transformer_path: str
    model_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictPipelineParams:
    with open(path, "r") as fin:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(fin))
