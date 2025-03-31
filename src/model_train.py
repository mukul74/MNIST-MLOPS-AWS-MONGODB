import logging
from typing import Dict, Tuple

import joblib
import numpy as np
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier

from src.connect_db import get_db

# Configure Logging
logging.basicConfig(level=logging.INFO)


# 🔹 Load Training Data from MongoDB
def load_data(db) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch training data from MongoDB."""
    try:
        train_collection = db["train_data"]
        train_data = list(
            train_collection.find({}, {"_id": 0, "features": 1, "label": 1})
        )  # Exclude `_id`

        if not train_data:
            raise HTTPException(
                status_code=404, detail="No training data found in the database"
            )

        X_train = np.array(
            [sample["features"] for sample in train_data], dtype=np.float32
        )
        y_train = np.array([sample["label"] for sample in train_data], dtype=np.int32)

        return X_train, y_train
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail="Data fetching unsuccessful")


# 🔹 Train Model
def train_model() -> Dict[str, str]:
    """Train a RandomForest model and save it."""
    try:
        db = get_db()
        X_train, y_train = load_data(db)

        if X_train.size == 0 or y_train.size == 0:
            raise HTTPException(status_code=404, detail="No valid training data found")
        logging.info("✅ Model training started")
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, verbose=1
        )
        model.fit(X_train, y_train)
        joblib.dump(model, "src/models/model.pkl")
        logging.info("✅ Model training successful")

        return {
            "status": "success",
            "message": "Model trained successfully",
            "code": 200,
        }
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail="Model training unsuccessful")
