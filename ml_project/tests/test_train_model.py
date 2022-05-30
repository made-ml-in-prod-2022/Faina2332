import pandas as pd
from typing import NoReturn, Tuple

from sklearn.ensemble import RandomForestClassifier

from src.models import train
from src.entities import TrainParams


def test_train_model(
    training_params: TrainParams,
    transformed_dataframe: Tuple[pd.Series, pd.DataFrame],
) -> NoReturn:
    target, transformed_dataset = transformed_dataframe
    model = train(transformed_dataset, target, training_params)
    params = model.get_params()

    assert isinstance(model, RandomForestClassifier)
    assert params['max_depth'] == 10
    assert params['n_estimators'] == 50
    assert params['criterion'] == 'gini'


