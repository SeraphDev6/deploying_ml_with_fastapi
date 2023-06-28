# Script to train machine learning model.
import yaml
from yaml import CLoader as Loader
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import process_data, train_model
# Add the necessary imports for the starter code.

with open("./params.yaml", "rb") as f:
    params = yaml.load(f, Loader=Loader)


# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb, scaler = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=False
)
# Train and save a model.
model = train_model(X_train,y_train)