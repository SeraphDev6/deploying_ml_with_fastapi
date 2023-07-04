from fastapi.testclient import TestClient
from pandas import read_csv
from starter.ml.helpers import cat_features
from main import app
from json import dumps
from random import choice, choices
client = TestClient(app)
data = read_csv("data/census.csv")
s_filter = data["salary"] == ">50K"
over_sample = data[s_filter].to_dict(orient="records")
under_sample = data[~s_filter].to_dict(orient="records")


def test_read_index():
    response = client.get("/")
    assert response.status_code == 200
    assert len(response.json()) == 5
    for key in ["Greeting",
                "Model_Info",
                "Model_Metrics",
                "Model_Metrics_Raw",
                "Privacy_Notice"]:
        assert key in response.json().keys()


def test_predict_one_over_50k():
    responses = []
    for _ in range(10):
        response = client.post("/predict_one?save_data=false",
                               content=dumps(choice(over_sample)))
        assert response.status_code == 200
        responses.append(response.json())
    assert all(map(lambda x: len(x) == 3, responses))
    assert all(map(lambda x: type(x) == dict, responses))
    assert any(map(lambda x: x["predicted"] == ">50K", responses))
    corrects = list(map(lambda x: x["correct"], responses))
    assert any(corrects)


def test_predict_one_under_50k():
    responses = []
    for _ in range(10):
        response = client.post("/predict_one?save_data=false",
                               content=dumps(choice(under_sample)))
        assert response.status_code == 200
        responses.append(response.json())
    assert all(map(lambda x: len(x) == 3, responses))
    assert all(map(lambda x: type(x) == dict, responses))
    assert any(map(lambda x: x["predicted"] == "<=50K", responses))
    corrects = list(map(lambda x: x["correct"], responses))
    assert any(corrects)


def test_predict_over_50k():
    response = client.post("/predict?save_data=false",
                           content=dumps({"inputs":
                                          choices(over_sample, k=10)}))
    assert response.status_code == 200
    assert len(response.json()["results"]) == 10
    assert any(map(lambda x: x["predicted"] == ">50K",
                   response.json()["results"]))
    assert any(map(lambda x: x["correct"], response.json()["results"]))


def test_predict_under_50k():
    response = client.post("/predict?save_data=false",
                           content=dumps({"inputs":
                                          choices(under_sample, k=10)}))
    assert response.status_code == 200
    assert len(response.json()["results"]) == 10
    assert any(map(lambda x: x["predicted"] == "<=50K",
                   response.json()["results"]))
    assert any(map(lambda x: x["correct"], response.json()["results"]))


def test_slice_metrics():
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
