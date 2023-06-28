from api import app
from json import load

@app.get("/")
async def index():
    model_results = load(open("model/eval.json"))
    model = model_results['model']
    metrics = model_results["results"]
    return {
        "Greeting": "Hello and welcome to this Machine Learning API!",
        "Model_Info": f"This API is used to host a {model} model \
which was trained on a set of census data \
to determine which income bracket an individual falls into.",
        "Model_Metrics": f"This model currently has a precision score of {metrics['precision']}, \
and a recall score of {metrics['recall']}, which gives us an fbeta score of \
{metrics['fbeta']} when calculated with a beta of 1",
"Model_Metrics_Raw": metrics,
            }

@app.post("/predict")
def predict():
    pass