from ml.helpers import load_model, load_params, load_eval
from reports.report import create_report
from ml.assessment import assess_slice_performance
import matplotlib.pyplot as plt
from pandas import DataFrame
from weasyprint import HTML

params = load_params() 
model = load_model()

def print_report(feature):
    prediction = assess_slice_performance(feature)
    prediction["values"].sort(key=lambda x: x["metrics"]["fbeta"]["vs_baseline"])
    baseline = load_eval()["metrics"]["fbeta"]

    data = DataFrame(map(lambda x: (x['name'], x['metrics']['fbeta']['value']),prediction["values"]))
    plt.figure(figsize=(20,10))
    plt.bar(data[0],data[1], label="slice")
    plt.xlabel(f"{feature} Unique Value")
    plt.ylabel("f1 Score")
    plt.axhline(y = baseline, color = 'r', linestyle = '-', label="baseline")
    plt.legend()
    plt.title("Slice Feature f1 score compared to baseline")
    plt.savefig(f"starter/reports/templates/images/{feature}_chart.png")
   
    html = create_report(prediction)
    HTML(string=html).write_pdf(f"starter/reports/slice_reports/{feature}_report.pdf")

