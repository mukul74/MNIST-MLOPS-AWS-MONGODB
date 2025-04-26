from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_docs():
    """Test the OPENAPI docs endpoint."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_home():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "FastAPI with MongoDB MNIST",
        "docs": "/docs",
        "redoc": "/redoc",
    }


def test_health():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_store_mnist_data():
    """Test the store MNIST data endpoint"""
    response = client.post("/store_mnist")
    assert response.status_code == 200

    json_response = response.json()
    assert json_response["status"] in ["success", "exists"]
    assert json_response["message"] in [
        "MNIST Data Stored in MongoDB",
        "MNIST Data already exists in MongoDB",
    ]
    assert json_response["code"] == 200


# def test_train_model():
#     """Test the train model endpoint"""
#     response = client.post("/train")
#     assert response.status_code == 200
#     assert response.json() == {
#         "status": "success",
#         "message": "Model trained, tuned, and logged successfully",
#         "code": 200,
#     }
