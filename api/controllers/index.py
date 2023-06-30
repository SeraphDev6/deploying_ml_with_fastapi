from fastapi import Request
from api import app
from api.helpers import recursive_get
from starter.ml.assessment import assess_slice_performance
from api.schemas import Prediciton, SortBy, Order, Column
from starter.ml.helpers import load_eval
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


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
                          f"and a recall score of {metrics['recall']}, which gives us an fbeta score of " +
                          f"{metrics['fbeta']} when calculated with a beta of 1"),
        "Model_Metrics_Raw": metrics,
            }


@app.post("/slice_metrics/", response_model=Prediciton)
async def predict(feature: Column, sort_by: SortBy = SortBy.num_records, order: Order = Order.descending):
    sort_by = sort_by.split(".")
    prediction = assess_slice_performance(feature)
    prediction["values"].sort(reverse=(order == "desc"),
                              key=lambda x: recursive_get(x, sort_by))
    return prediction

app.mount("/static", StaticFiles(directory="starter/reports/templates/images"), name="static")
templates = Jinja2Templates(directory="starter/reports/templates/")


@app.get("/report/{feature}", response_class=HTMLResponse)
async def report(request: Request, feature: Column):
    data = await predict(feature, SortBy.fbeta, Order.ascending)

    return templates.TemplateResponse("web_template.html",
                                      {"request": request,
                                       "data": data,
                                       "options": [x.value for x in Column]})
