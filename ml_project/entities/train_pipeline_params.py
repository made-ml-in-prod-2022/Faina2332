from dataclasses import dataclass
from .split_params import SplitParams
from .feature_params import FeatureParams
from .train_params import TrainParams
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainPipelineParams:
    """ Defines train pipeline parameters. """
    input_data_path: str
    output_model_path: str
    metrics_path: str
    output_transformer_path: str
    splitting_params: SplitParams
    feature_params: FeatureParams
    train_params: TrainParams


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
