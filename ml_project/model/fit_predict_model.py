from typing import Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from ml_project.entities.train_params import TrainParams


def train(features: pd.DataFrame, target: pd.Series,
          train_params: TrainParams) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=train_params.n_estimators,
        max_depth=train_params.max_depth,
        random_state=train_params.random_state
    )

    model.fit(features, target)
    return model


def predict(model: RandomForestClassifier, features: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(features)
    return predictions


def evaluate(predictions: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "ROC_AUC": roc_auc_score(target, predictions),
        "Accuracy": accuracy_score(target, predictions),
        "F1": f1_score(target, predictions)
    }
