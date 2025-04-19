import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.model_predict import predict_digit, predict_sample
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


@app.post("/predict_image", summary="Predict MNIST Image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    arr = np.array(image).astype("float32") / 255.0
    response = predict_digit(arr)
    # print("Predcited_Response : ", int(response["predicted_digit"]))
    return {
        "status": "success",
        "message": "Prediction successful",
        "predicted_digit": int(response["predicted_digit"]),
        "code": 200,
    }


@app.get("/debug-env")
def debug_env():
    import os

    return {
        "RUNNING_IN_DOCKER": os.getenv("RUNNING_IN_DOCKER"),
        "MONGO_URI": os.getenv("MONGO_URI"),
        "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME"),
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
