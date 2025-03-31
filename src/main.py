import logging

from fastapi import FastAPI, HTTPException

from src.model_predict import predict_sample
from src.model_train import train_model
from src.store_minst import store_mnist_model

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="MNIST FastAPI Service",
    description="A simple FastAPI service to store, train, and predict MNIST data using MongoDB",
    version="1.0.0",
)


@app.post("/store_mnist", summary="Store MNIST Data")
async def store_mnist_data():
    """Endpoint to store MNIST data in MongoDB."""
    try:
        logging.info("📥 Storing MNIST data...")
        response = store_mnist_model()
        logging.info("✅ MNIST data stored successfully")
        return response
    except Exception as e:
        logging.error(f"❌ Error storing MNIST data: {e}")
        raise HTTPException(status_code=500, detail="Failed to store MNIST data")


@app.get("/", summary="API Root")
def home():
    """Root endpoint with API metadata."""
    return {"message": "FastAPI with MongoDB MNIST", "docs": "/docs", "redoc": "/redoc"}


@app.post("/train", summary="Train Model")
def train():
    """Endpoint to train the MNIST model."""
    try:
        logging.info("🚀 Training model...")
        response = train_model()
        logging.info("✅ Model trained successfully")
        return response
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise HTTPException(status_code=500, detail="Model training failed")


@app.get("/predict_sample/{n}", summary="Predict MNIST Sample")
def predict(n: int = 5):
    """Endpoint to predict n test samples."""
    try:
        logging.info(f"🔍 Making predictions for {n} samples...")
        response = predict_sample(n)
        logging.info("✅ Prediction successful")
        return response
    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
