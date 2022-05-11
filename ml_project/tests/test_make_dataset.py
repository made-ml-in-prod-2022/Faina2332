import pandas as pd

from typing import NoReturn

from entities import SplitParams
from data import split_train_val_data


def test_split_train_val_data(data: pd.DataFrame) -> NoReturn:
    test_size = SplitParams.val_size
    random_seed = SplitParams.random_state
    params = SplitParams(val_size=test_size, random_state=random_seed)
    train_df, test_df = split_train_val_data(data, params)

    assert test_df.shape[0] <= data.shape[0] * test_size
    assert train_df.shape[0] + test_df.shape[0] == data.shape[0]

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
