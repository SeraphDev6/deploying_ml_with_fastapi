"""
Data Preparation Stage for DVC pipeline
"""
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.helpers import load_params, cat_features


params = load_params()
data = pd.read_csv("data/census.csv")
train, test = train_test_split(data, test_size=0.20)


X_train, y_train, encoder, lb, scaler = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb, scaler=scaler
)

np.savetxt("data/train/X.csv", X_train)
np.savetxt("data/train/y.csv", y_train)
np.savetxt("data/test/X.csv", X_test)
np.savetxt("data/test/y.csv", y_test)
dump(encoder, "data/preprocessors/encoder.pkl")
dump(lb, "data/preprocessors/lb.pkl")
dump(scaler, "data/preprocessors/scaler.pkl")
