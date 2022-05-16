import pytest
from faker import Faker
import pandas as pd
from typing import List, Tuple


from ml_project.src.entities import FeatureParams, TrainParams
from ml_project.src.features import column_transformer, extract_target


@pytest.fixture(scope="session")
def n_rows() -> int:
    return 200


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
def fake_data(n_rows) -> pd.DataFrame:
    faker = Faker()
    Faker.seed(42)
    fake_data = {
        "age": [faker.pyint(min_value=27, max_value=79) for _ in range(n_rows)],
        "sex": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)],
        "cp": [faker.pyint(min_value=0, max_value=3) for _ in range(n_rows)],
        "trestbps": [faker.pyint(min_value=89, max_value=205) for _ in range(n_rows)],
        "chol": [faker.pyint(min_value=104, max_value=586) for _ in range(n_rows)],
        "fbs": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)],
        "restecg": [faker.pyint(min_value=0, max_value=2) for _ in range(n_rows)],
        "thalach": [faker.pyint(min_value=64, max_value=209) for _ in range(n_rows)],
        "exang": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)],
        "oldpeak": [faker.pyfloat(min_value=0, max_value=6.2) for _ in range(n_rows)],
        "slope": [faker.pyint(min_value=0, max_value=2) for _ in range(n_rows)],
        "ca": [faker.pyint(min_value=0, max_value=4) for _ in range(n_rows)],
        "thal": [faker.pyint(min_value=0, max_value=3) for _ in range(n_rows)],
        "condition": [faker.pyint(min_value=0, max_value=1) for _ in range(n_rows)]
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
def training_params() -> TrainParams:
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

    train_features, train_target = extract_target(fake_data, feature_params)
    transformer = column_transformer(feature_params)
    transformer.fit(train_features)
    transformed_features = transformer.transform(train_features)

    return train_target, transformed_features


