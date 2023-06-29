# Script to train machine learning model.
from joblib import dump
from ml import train_model
from ml.helpers import load_train_data, load_params
# Add the necessary imports for the starter code.


params = load_params()
X_train, y_train = load_train_data()

# Train and save a model.
model = train_model(X_train, y_train, params=params["tuning"],
                    gridsearch=params["gridsearch"],
                    model_name=params["model"])

dump(model, f"model/{params['model']}.pkl")
