# Script to train machine learning model.
import yaml
import numpy as np
from yaml import CLoader as Loader
from joblib import dump
from ml import train_model
# Add the necessary imports for the starter code.

with open("./params.yaml", "rb") as f:
    params = yaml.load(f, Loader=Loader)

X_train = np.loadtxt("data/train/X.csv")
y_train = np.loadtxt("data/train/y.csv")

# Train and save a model.
model = train_model(X_train,y_train,params=params["tuning"], gridsearch=params["gridsearch"])

dump(model,"model/RandomForestClassifier.pkl")