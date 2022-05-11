import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.entities.feature_params import FeatureParams


def cat_features_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    encoder = OneHotEncoder()
    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("encoder", encoder),
        ]
    )
    return pipeline


def num_features_pipeline() -> Pipeline:
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    scaler = StandardScaler()

    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", scaler),
        ]
    )
    return pipeline


def column_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "cat",
                cat_features_pipeline(),
                params.categorical_features,
            ),
            (
                "num",
                num_features_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
