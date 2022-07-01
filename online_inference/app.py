import logging
import os
import pickle
from typing import List, Union

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from fastapi import HTTPException
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model = None
transformer = None
app = FastAPI()


class HeartDiseaseModel(BaseModel):
    data: List[List[int]]
    features: List[str]


class ModelResponse(BaseModel):
    condition: int


def make_predict(
        data: List,
        features: List[str],
        model: RandomForestClassifier,
        transformer: ColumnTransformer,
) -> List[ModelResponse]:

    if len(data) == 0:
        raise HTTPException(
            status_code=400, detail="Empty input"
        )

    data = pd.DataFrame(data, columns=features)
    transfromed_data = pd.DataFrame(transformer.transform(data))
    predictions = model.predict(transfromed_data)
    return [ModelResponse(condition=int(disease)) for disease in predictions]


def load_object(path: str) -> Union[RandomForestClassifier, ColumnTransformer]:
    with open(path, "rb") as f:
        return pickle.load(f)


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")

    if model_path is None:
        err = f"PATH_TO_MODEL is {model_path}"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)

@app.on_event("startup")
def load_transformer():
    global transformer
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")

    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER is {transformer_path}"
        logger.error(err)
        raise RuntimeError(err)

    transformer = load_object(transformer_path)

@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict", response_model=List[ModelResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model, transformer)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))