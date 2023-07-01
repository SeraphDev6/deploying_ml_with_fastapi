from joblib import load
import yaml
from numpy import loadtxt
from yaml import CLoader as Loader
from json import load as load_json
from .data import process_data
from .model import inference
from pandas import DataFrame


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


def predict_on_raw(raw_df):
    not_available = raw_df["salary"].isna()
    raw_df["salary"].fillna("<=50K", inplace=True)
    X, y, _, _, _ = process_data(raw_df, cat_features, "salary",
                                 False, *load_processors())
    output = DataFrame(y, columns=["actual"])
    output["actual"][not_available] = -1
    output["predicted"] = inference(load_model(), X)
    output["correct"] = output["actual"] == output["predicted"]
    return output
