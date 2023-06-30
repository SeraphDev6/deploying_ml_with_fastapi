from fastapi.testclient import TestClient
from pandas import read_csv
from starter.ml.helpers import cat_features
from main import app

client = TestClient(app)


def test_read_index():
    response = client.get("/")
    assert response.status_code == 200
    assert len(response.json()) == 4
    for key in ["Greeting",
                "Model_Info",
                "Model_Metrics",
                "Model_Metrics_Raw"]:
        assert key in response.json().keys()


def test_slice_metrics():
    data = read_csv("data/census.csv")
    for column in cat_features:
        response = client.post(f"/slice_metrics?feature={column}",
                               headers={"accept": "application/json"})
        assert response.status_code == 200
        prediction = response.json()
        assert ["feature", "values"] == list(prediction.keys())
        values = prediction["values"]
        unique_vals = data[column].unique()
        for val in values:
            assert val["name"] in unique_vals
            assert val["num_records"] >= 0
            assert len(val["metrics"]) == 3


def test_invalid_path():
    response = client.get("/not_valid")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_html_page():
    for column in cat_features:
        response = client.get(f"/report/{column}")
        assert response.status_code == 200
        assert "html" in response.text
    response = client.get("/report/not_a_column")
    assert response.status_code == 422
