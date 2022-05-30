import pandas as pd

from typing import NoReturn

from src.data import split_train_val_data
from src.entities import SplitParams


def test_split_train_val_data(fake_data: pd.DataFrame) -> NoReturn:
    test_size = 0.2
    random_seed = 42
    params = SplitParams(val_size=test_size, random_state=random_seed)
    train_df, test_df = split_train_val_data(fake_data, params)

    assert test_df.shape[0] <= fake_data.shape[0] * test_size
    assert train_df.shape[0] + test_df.shape[0] == fake_data.shape[0]

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
