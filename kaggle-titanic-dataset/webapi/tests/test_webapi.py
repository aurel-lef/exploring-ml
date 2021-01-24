from app.main import app
from fastapi.testclient import TestClient


def test_webapi():
    client = TestClient(app)
    train = '../../data/train.csv'
    test = '../../data/test.csv'

    response = client.post(
        "/predict",
        files={"file": ("test.csv", open(test, 'rb'), "text/csv")}
    )
    assert response.status_code == 400
    assert "not trained" in response.json()["detail"]

    response = client.post(
        "/train",
        files={"file": ("train.csv", open(train, 'rb'), "text/csv")}
    )
    assert response.status_code == 200
    response = client.post(
        "/predict",
        files={"file": ("test.csv", open(test, 'rb'), "text/csv")}
    )
    assert response.status_code == 200
