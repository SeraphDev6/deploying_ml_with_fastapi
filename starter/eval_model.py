from json import dump
from ml.helpers import (load_params,
                        load_test_data,
                        load_model,
                        load_train_data)
from ml.model import (compute_metrics_from_model,
                              compute_model_metrics,
                              inference)

params = load_params()
X_train, y_train = load_train_data()
X_test, y_test = load_test_data()
model = load_model()

predictions = inference(model, X_test)

labels = ("precision", "recall", "fbeta")
values = compute_model_metrics(y_test, predictions)
train_values = compute_metrics_from_model(model, X_train, y_train)
results = {"model": params["model"],
           "metrics": dict(zip(labels, values)),
           "training_metrics": dict(zip(labels, train_values))}
with open("model/eval.json", "w") as f:
    dump(results, f)
