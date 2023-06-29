from sklearn.base import ClassifierMixin
from starter.ml.helpers import load_eval, load_model, load_test_data
from starter.ml.model import compute_metrics_from_model, inference
from numpy import ndarray


def test_loading_model():
    model = load_model()
    assert isinstance(model, ClassifierMixin)


def test_inference():
    X_test, _ = load_test_data()
    predictions = inference(load_model(), X_test)
    assert isinstance(predictions, ndarray)
    assert predictions.max() <= 1
    assert predictions.min() >= 0
    assert predictions.shape[0] == X_test.shape[0]


def test_metrics():
    X_test, y_test = load_test_data()
    metrics = compute_metrics_from_model(load_model(), X_test, y_test)
    assert isinstance(metrics, tuple)
    assert len(metrics) == 3
    baseline = tuple(load_eval()["metrics"].values())
    print(metrics)
    print(baseline)
    for i, metric in enumerate(metrics):
        assert abs(metric - baseline[i]) <= 0.05
