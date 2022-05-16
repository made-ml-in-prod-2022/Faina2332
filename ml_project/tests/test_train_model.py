import pandas as pd
from typing import NoReturn, Tuple

from sklearn.ensemble import RandomForestClassifier

from ml_project.src.models import train
from ml_project.src.entities import TrainParams


def test_train_model(
    training_params: TrainParams,
    transformed_dataframe: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_dataset = transformed_dataframe
    model = train(transformed_dataset, target, training_params)
    assert isinstance(model, RandomForestClassifier)


