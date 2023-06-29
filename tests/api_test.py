from fastapi.testclient import TestClient

from api import app
client = TestClient(app)

def test_read_index():
    response = client.get("/")
    assert response.status_code == 200