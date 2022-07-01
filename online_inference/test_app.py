from fastapi.testclient import TestClient

from app import app
import pytest

from app import HeartDiseaseModel

@pytest.fixture()
def get_test_data():
    data = [
        HeartDiseaseModel(
            data = [[69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0]],
            features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        )
    ]
    return data

def test_predict(get_test_data):
    with TestClient(app) as client:
        responce_data = get_test_data[0]
        test_data = responce_data.data
        features = responce_data.features
        response = client.get("/predict", json={"data": test_data, "features": features})

        assert response.status_code == 200
        assert response.json() == [{"condition": 0}]


def test_load_on_start():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_predict_extreme_data_value(get_test_data):
    with TestClient(app) as client:
        test_data = 0
        features = 0
        response = client.get("/predict", json={"data": test_data, "features": features})
        assert response.status_code == 422


def test_predict_request_no_data(get_test_data):
    with TestClient(app) as client:
        responce_data = get_test_data[0]
        features = responce_data.features
        response = client.get("/predict", json={"data": [], "features": features})
        assert response.status_code == 400
        assert "Empty input" in response.json()["detail"]