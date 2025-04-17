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


def test_predict_endpoint():
    """Test the prediction endpoint"""
    response = client.get("/predict_sample/5")
    assert response.status_code == 200

    json_response = response.json()

    assert json_response["status"] == "success"
    assert json_response["message"] == "Prediction succesfull"
    assert json_response["code"] == 200

    # Ensure actual and predicted are lists of the same length
    actual = json_response["actual"]
    predicted = json_response["predicted"]

    assert isinstance(actual, list)
    assert isinstance(predicted, list)
    assert len(actual) == len(predicted)

    # Optionally check if values are integers between 0 and 9 (for MNIST)
    assert all(isinstance(x, int) and 0 <= x <= 9 for x in actual)
    assert all(isinstance(x, int) and 0 <= x <= 9 for x in predicted)


def test_train_model():
    """Test the train model endpoint"""
    response = client.post("/train")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model trained successfully",
        "code": 200,
    }
