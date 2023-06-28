import numpy as np
import json
from joblib import load
import yaml
from ml import compute_model_metrics, inference
from yaml import CLoader as Loader

with open("./params.yaml", "rb") as f:
    params = yaml.load(f, Loader=Loader)

X_test = np.loadtxt("data/test/X.csv")
y_test = np.loadtxt("data/test/y.csv")
model = load(f"model/{params['model']}.pkl")

predictions = inference(model, X_test)

labels = ("precision", "recall", "fbeta")
values = compute_model_metrics(y_test, predictions)
results = {"model": params["model"], "results": dict(zip(labels,values))}
with open("model/eval.json","w") as f:
    json.dump(results,f)