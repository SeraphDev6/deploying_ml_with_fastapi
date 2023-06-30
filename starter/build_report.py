import os
from ml.helpers import load_eval, cat_features
from reports.report import create_report_inner, create_report_full
from ml.assessment import assess_slice_performance
import matplotlib.pyplot as plt
from pandas import DataFrame

def create_chart(values, feature):
    baseline = load_eval()["metrics"]["fbeta"]
    data = DataFrame(map(lambda x: (x['name'], x['metrics']['fbeta']['value']),values))
    plt.figure(figsize=(20,15))
    plt.rcParams.update({'font.size': 16})
    plt.bar(data[0],data[1], label="slice")
    plt.xlabel(f"{feature} Unique Value")
    plt.xticks(rotation=90)
    plt.ylabel("f1 Score")
    plt.axhline(y = baseline, color = 'r', linestyle = '-', label="baseline")
    plt.legend()
    plt.title("Slice Feature f1 score compared to baseline")
    plt.savefig(f"starter/reports/templates/images/{feature}_chart.png")

def print_report(features):
    inner = ""
    for feature in features:
        prediction = assess_slice_performance(feature)
        prediction["values"].sort(key=lambda x: x["metrics"]["fbeta"]["vs_baseline"])
        create_chart(prediction["values"], feature)
        inner += create_report_inner(prediction) 
    with open(f"starter/reports/slice_report.html","w") as f:
        html = create_report_full(inner)
        f.write(html)

os.mkdir("starter/reports/templates/images")
print_report(cat_features)
