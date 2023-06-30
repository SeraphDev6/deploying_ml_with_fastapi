import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('starter/reports/templates'))
template = env.get_template('report_template.html')

def create_report(data):
    html =  template.render(data=data)
    return html