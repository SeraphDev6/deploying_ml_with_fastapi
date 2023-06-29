from .helpers import load_eval, load_model, cat_features, load_processors
from .data import process_data
from pandas import read_csv
from .model import compute_metrics_from_model

encoder, lb, scaler = load_processors()
baseline = load_eval()["metrics"]
data = read_csv("data/census.csv")


def assess_slice_performance(column_name, model=load_model(), data=data):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    data : pd.DataFrame
        Data used for prediction.
    column_name: str
        Column to slice the data by
    Returns
    -------
    results: dict
        A multi layered dictionary which contains each unique value as a key,
        with the three metrics siplayed in the sub dictionary
    """
    results = {"feature": column_name, "values": [], "vs_baseline": []}
    for val in data[column_name].unique():
        data_slice = data.loc[data[column_name] == val, :]
        X, y, _, _, _ = process_data(data_slice, cat_features, "salary",
                                     False, encoder, lb, scaler)
        value = {"name": val,
                 "num_records": X.shape[0],
                 "metrics": {}}
        labels = ("precision", "recall", "fbeta")
        metrics = compute_metrics_from_model(model, X, y)
        for i, label in enumerate(labels):
            value["metrics"][label] = {"value": metrics[i],
                                       "vs_baseline": (metrics[i] -
                                                       baseline[label])}
        results["values"].append(value)
    return results
