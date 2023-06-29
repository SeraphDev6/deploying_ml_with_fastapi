from api import app
from api.helpers import recursive_get
from starter.ml import assess_slice_performance
from api.schemas import Prediciton, SortBy, Order, Column
from starter.ml.helpers import load_eval


@app.get("/")
async def index():
    model_results = load_eval()
    model = model_results['model']
    metrics = model_results["metrics"]
    return {
        "Greeting": "Hello and welcome to this Machine Learning API!",
        "Model_Info": (f"This API is used to host a {model} model " +
                       "which was trained on a set of census data " +
                       "to determine which income bracket an individual falls into."),
        "Model_Metrics": (f"This model currently has a precision score of {metrics['precision']}, " +
                          "and a recall score of {metrics['recall']}, which gives us an fbeta score of " +
                          "{metrics['fbeta']} when calculated with a beta of 1"),
        "Model_Metrics_Raw": metrics,
            }


@app.post("/slice_metrics/", response_model=Prediciton)
async def predict(feature: Column, sort_by: SortBy = SortBy.num_records, order: Order = Order.descending):
    sort_by = sort_by.split(".")
    prediction = assess_slice_performance(feature)
    prediction["values"].sort(reverse=(order == "desc"),
                              key=lambda x: recursive_get(x, sort_by))
    return prediction
