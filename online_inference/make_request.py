import pandas as pd
import requests


def predict():
    data = pd.read_csv("data/data.csv")
    request_features = list(data.columns)
    print(request_features)
    for i in range(data.shape[0]):
        request_data = [int(x) for x in data.iloc[i].tolist()]

        print(request_data)
        response = requests.get(
            "http://localhost:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())


if __name__ == "__main__":
    predict()