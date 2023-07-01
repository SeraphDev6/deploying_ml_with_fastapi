from pydantic import BaseModel
from enum import Enum


class Column(str, Enum):
    workclass = "workclass",
    education = "education",
    marital_status = "marital-status",
    occupation = "occupation",
    relationship = "relationship",
    race = "race",
    sex = "sex",
    native_country = "native-country",


class SortBy(str, Enum):
    num_records = "num_records"
    precision = "metrics.precision.value"
    recall = "metrics.recall.value"
    fbeta = "metrics.fbeta.value"


class Order(str, Enum):
    ascending = "asc"
    descending = "desc"


class MetricScore(BaseModel):
    value: float
    vs_baseline: float


class Metrics(BaseModel):
    precision: MetricScore
    recall: MetricScore
    fbeta: MetricScore


class Value(BaseModel):
    name: str
    num_records: int
    metrics: Metrics


class FullMetrics(BaseModel):
    feature: Column
    values: list[Value]
