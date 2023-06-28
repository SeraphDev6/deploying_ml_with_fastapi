import numpy as np
import json
from joblib import load
from ml import compute_model_metrics, inference

X_test = np.loadtxt("data/test/X.csv")
y_test = np.loadtxt("data/test/y.csv")
model = load("model/RandomForestClassifier.pkl")

predictions = inference(model, X_test)

labels = ("precision", "recall", "fbeta")
values = compute_model_metrics(y_test, predictions)

with open("model/eval.json","w") as f:
    json.dump(dict(zip(labels,values)),f)