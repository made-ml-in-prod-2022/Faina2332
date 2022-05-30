from fastapi.testclient import TestClient

from app import app


def test_predict():
    with TestClient(app) as client:
        test_data = [[69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0]]
        features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

        response = client.get("/predict", json={"data": test_data, "features": features})

        assert response.status_code == 200
        assert response.json() == [{"condition": 0}]


def test_load_on_start():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200

