from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_val_data(
        data: pd.DataFrame, params
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
