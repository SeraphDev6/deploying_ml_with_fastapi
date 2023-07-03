import requests
from random import choices
from pandas import read_csv
API_URL = "https://fast-ml-api.fly.dev/"

def post_to_live_api():
    data = choices(read_csv("data/census.csv").to_dict(orient="records"), k=10)
    results = requests.post(API_URL+"predict",json={"inputs": data})
    return results.status_code, results.json()

if __name__ == "__main__":
    print(post_to_live_api())