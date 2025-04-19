import logging

import joblib
import numpy as np
from fastapi import HTTPException

from src.connect_db import get_db

# Configure Logging
logging.basicConfig(level=logging.INFO)


def predict_digit(image: np.ndarray) -> dict:
    """Predict the digit from the image using the trained model."""
    try:
        # Load model with proper error handling
        try:
            model = joblib.load("src/models/model.pkl")
        except FileNotFoundError:
            logging.error("Trained model not found")
            raise HTTPException(status_code=500, detail="Trained model not found")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Error loading trained model")
        # pdb.set_trace()

        # Preprocess image
        image = (image * 255).clip(0, 255).astype(np.uint8)
        print(image)
        image = image.reshape(1, -1).astype(np.float32)

        # Perform prediction
        prediction = model.predict(image)
        logging.info("✅ Prediction successful")

        return {
            "status": "success",
            "message": "Prediction successful",
            "predicted_digit": int(prediction),
            "code": 200,
        }

    except HTTPException as http_err:
        raise http_err  # Directly propagate HTTPExceptions
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction unsuccessful")


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
        print(X_test[0])
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
        print(X_test.shape)

        predictions = model.predict(X_test)
        logging.info("✅ Prediction successful")

        return {
            "status": "success",
            "message": "Prediction succesfull",
            "actual": y_test.tolist(),
            "predicted": predictions.tolist(),
            "code": 200,
        }

    except HTTPException as http_err:
        raise http_err  # Directly propagate HTTPExceptions
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction unsuccessful")
