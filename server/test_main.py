from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to FastAPI on Railway!"}

def test_read_item():
    response = client.get("/items/1?q=test")
    assert response.status_code == 200
    assert response.json() == {"item_id": 1, "q": "test"}