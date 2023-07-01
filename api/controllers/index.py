from fastapi import Request
from api import app
from api.helpers import recursive_get
from starter.ml.assessment import assess_slice_performance
from api.schemas import (FullMetrics,
                         SortBy,
                         Order,
                         Column,
                         Input,
                         ListOutput,
                         Salary,
                         Output,
                         MassInput)
from starter.ml.helpers import load_eval, predict_on_raw
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
        "Privacy_Notice": ("If using the inference paths, /predict and /predict_one, the API will save " +
                           "the inputed data by default. To prevent this add the query option " +
                           "save_data=false, (ex. /predict?save_data=false). By using this API you are " +
                           "consenting to the API saving any data entered")}


@app.post("/predict_one", response_model=Output)
async def predict_one(input: Input,
                      save_data: bool = True):
    df = input.to_df()
    results = predict_on_raw(df)
    return process_input(df, results, save_data)[0]


@app.post("/predict", response_model=ListOutput)
async def predict_many(input: MassInput,
                       save_data: bool = True):
    df = input.to_df()
    results = predict_on_raw(df)
    return {"results": process_input(df, results, save_data)}


@app.post("/slice_metrics/", response_model=FullMetrics)
async def predict(feature: Column,
                  sort_by: SortBy = SortBy.num_records,
                  order: Order = Order.descending):
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


def process_input(df, results, save_data):
    df["salary"] = results["actual"].apply(
        lambda x: [s.value for s in Salary][int(x)] if x >= 0 else None)
    df["salary_prdiction"] = results["predicted"].apply(
        lambda x: [s.value for s in Salary][int(x)])
    results["predicted"] = results["predicted"].apply(
        lambda x: [s.value for s in Salary][int(x)])
    results["actual"] = results["actual"].apply(
        lambda x: [s.value for s in Salary][int(x)] if x >= 0 else None)
    results.loc[results["actual"].isna(), "correct"] = None
    if save_data:
        with open("data/user_input.csv", "a") as f:
            f.write(df.to_csv(index=False, header=False)[:-1])
    return results.to_dict(orient="records")
