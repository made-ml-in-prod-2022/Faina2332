import pytest
from faker import Faker

from typing import List, Tuple

import pandas as pd

from src.entities import FeatureParams, TrainParams, TrainPipelineParams, SplitParams, PredictPipelineParams
from src.features import column_transformer, extract_target
from src.train_pipeline import train_pipeline

N_ROWS = 300


@pytest.fixture(scope="session")
def test_data_path() -> str:
    return "tests/test_data/test_data.csv"


@pytest.fixture(scope="session")
def output_predictions_path() -> str:
    return "tests/test_data/test_predictions.csv"


@pytest.fixture(scope="session")
def load_model_path() -> str:
    return "tests/test_data/test_model.pkl"


@pytest.fixture(scope="session")
def metrics_path() -> str:
    return "tests/test_data/test_metrics.json"


@pytest.fixture(scope="session")
def load_transformer_path() -> str:
    return "tests/test_data/test_transformer.pkl"


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="session")
def target_col() -> str:
    return "condition"


@pytest.fixture(scope="session")
def fake_data() -> pd.DataFrame:
    faker = Faker()
    Faker.seed(42)
    fake_data = {
        "age": [faker.pyint(min_value=27, max_value=79) for _ in range(N_ROWS)],
        "sex": [faker.pyint(min_value=0, max_value=1) for _ in range(N_ROWS)],
        "cp": [faker.pyint(min_value=0, max_value=3) for _ in range(N_ROWS)],
        "trestbps": [faker.pyint(min_value=89, max_value=205) for _ in range(N_ROWS)],
        "chol": [faker.pyint(min_value=104, max_value=586) for _ in range(N_ROWS)],
        "fbs": [faker.pyint(min_value=0, max_value=1) for _ in range(N_ROWS)],
        "restecg": [faker.pyint(min_value=0, max_value=2) for _ in range(N_ROWS)],
        "thalach": [faker.pyint(min_value=64, max_value=209) for _ in range(N_ROWS)],
        "exang": [faker.pyint(min_value=0, max_value=1) for _ in range(N_ROWS)],
        "oldpeak": [faker.pyfloat(min_value=0, max_value=6.2) for _ in range(N_ROWS)],
        "slope": [faker.pyint(min_value=0, max_value=2) for _ in range(N_ROWS)],
        "ca": [faker.pyint(min_value=0, max_value=4) for _ in range(N_ROWS)],
        "thal": [faker.pyint(min_value=0, max_value=3) for _ in range(N_ROWS)],
        "condition": [faker.pyint(min_value=0, max_value=1) for _ in range(N_ROWS)]
    }
    return pd.DataFrame(data=fake_data)


@pytest.fixture(scope="session")
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str
) -> FeatureParams:
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col
    )
    return feature_params


@pytest.fixture(scope="package")
def random_forest_training_params() -> TrainParams:
    model = TrainParams(
        model_type="RandomForestClassifier",
        n_estimators=50,
        max_depth=10,
        random_state=42
    )
    return model


@pytest.fixture(scope="package")
def transformed_dataframe(
        fake_data: pd.DataFrame,
        feature_params: FeatureParams
) -> Tuple[pd.Series, pd.DataFrame]:
    transformer = column_transformer(feature_params)
    transformer.fit(fake_data)

    transformed_features = transformer.transform(fake_data)
    target = extract_target(fake_data, feature_params)

    return target, transformed_features


@pytest.fixture(scope="package")
def train_pipeline_params(
        fake_data_path: str,
        load_model_path: str,
        metrics_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        target_col: str,
        load_transformer_path: str,
        random_forest_training_params: TrainParams
) -> TrainPipelineParams:
    train_pipeline_params = TrainPipelineParams(
        input_data_path=fake_data_path,
        metrics_path=metrics_path,
        output_model_path=load_model_path,
        output_transformer_path=load_transformer_path,
        splitting_params=SplitParams(val_size=0.2, random_state=42),
        feature_params=FeatureParams(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            target_col=target_col
        ),
        train_params=random_forest_training_params
    )
    return train_pipeline_params


@pytest.fixture(scope="package")
def predict_pipeline_params(
        fake_data_path: str,
        load_model_path: str,
        output_predictions_path: str,
        load_transformer_path: str,
) -> PredictPipelineParams:
    pred_pipeline_params = PredictPipelineParams(
        input_data_path=fake_data_path,
        predict_path=output_predictions_path,
        transformer_path=load_transformer_path,
        model_path=load_model_path,
    )
    return pred_pipeline_params


@pytest.fixture(scope="package")
def train_on_fake_data(config_path: str):
    train_pipeline(config_path)
