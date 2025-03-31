import logging

import joblib
import numpy as np
from fastapi import HTTPException

from src.connect_db import get_db

# Configure Logging
logging.basicConfig(level=logging.INFO)


# 🔹 Fetch Test Sample & Predict
def predict_sample(n: int = 5):
    """Fetch test samples from MongoDB and make predictions."""
    try:
        db = get_db()
        test_collection = db["test_data"]

        # Query test data with projection
        test_data = list(
            test_collection.find({}, {"_id": 0, "features": 1, "label": 1}).limit(n)
        )
        if not test_data:
            raise HTTPException(status_code=404, detail="No test data found")

        X_test = np.array(
            [sample["features"] for sample in test_data], dtype=np.float32
        )
        y_test = np.array([sample["label"] for sample in test_data], dtype=np.int32)

        # Load model with proper error handling
        try:
            model = joblib.load("src/models/model.pkl")
        except FileNotFoundError:
            logging.error("Trained model not found")
            raise HTTPException(status_code=500, detail="Trained model not found")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Error loading trained model")

        # Perform prediction
        predictions = model.predict(X_test)
        logging.info("✅ Prediction successful")

        return {
            "message": "Prediction Done",
            "actual": y_test.tolist(),
            "predicted": predictions.tolist(),
        }

    except HTTPException as http_err:
        raise http_err  # Directly propagate HTTPExceptions
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction unsuccessful")
