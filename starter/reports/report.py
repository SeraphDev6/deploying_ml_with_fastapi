from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('starter/reports/templates'))
inner = env.get_template('report_inner.html')
outer = env.get_template('report_outer.html')


def create_report_inner(data):
    return inner.render(data=data)


def create_report_full(inner_html):
    return outer.render(inner=inner_html)
