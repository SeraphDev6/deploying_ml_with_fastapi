from joblib import load
import yaml
from numpy import loadtxt
from yaml import CLoader as Loader
from json import load as load_json


def load_params():
    with open("params.yaml", "rb") as f:
        params = yaml.load(f, Loader=Loader)
    return params


def load_model():
    return load(f"model/{load_params()['model']}.pkl")


def load_processors():
    encoder = load("data/preprocessors/encoder.pkl")
    lb = load("data/preprocessors/lb.pkl")
    scaler = load("data/preprocessors/scaler.pkl")
    return encoder, lb, scaler


def load_train_data():
    X_train = loadtxt("data/train/X.csv")
    y_train = loadtxt("data/train/y.csv")
    return X_train, y_train


def load_test_data():
    X_test = loadtxt("data/test/X.csv")
    y_test = loadtxt("data/test/y.csv")
    return X_test, y_test


def load_eval():
    return load_json(open("model/eval.json"))


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
    ]
