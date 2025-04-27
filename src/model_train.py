import logging
import os
import tempfile
from typing import Dict, Tuple

import joblib
import mlflow
import numpy as np
import optuna
from fastapi import HTTPException
from scipy.ndimage import rotate, shift, zoom
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.connect_db import get_db

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Set MLflow Experiment
mlflow.set_experiment("MyFastAPIApp_MNIST_Experiment")


def load_data(db) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch training and test data from MongoDB."""
    try:
        train_collection = db["train_data"]
        test_collection = db["test_data"]

        train_data = list(
            train_collection.find({}, {"_id": 0, "features": 1, "label": 1})
        )
        test_data = list(
            test_collection.find({}, {"_id": 0, "features": 1, "label": 1})
        )

        if not train_data:
            raise HTTPException(
                status_code=404, detail="No training data found in the database"
            )
        if not test_data:
            raise HTTPException(
                status_code=404, detail="No test data found in the database"
            )

        X_train = np.array(
            [sample["features"] for sample in train_data], dtype=np.float32
        )
        y_train = np.array([sample["label"] for sample in train_data], dtype=np.int32)
        X_test = np.array(
            [sample["features"] for sample in test_data], dtype=np.float32
        )
        y_test = np.array([sample["label"] for sample in test_data], dtype=np.int32)

        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail="Data fetching unsuccessful")


def augment_data(X, y):
    """Apply random shifts, rotations, scaling, and noise to the dataset."""
    X_augmented = []
    y_augmented = []

    for img, label in zip(X, y):
        img = img.reshape(28, 28)  # Assuming images are flattened 28x28

        # Random shift
        shift_x = np.random.uniform(-2, 2)
        shift_y = np.random.uniform(-2, 2)
        shifted = shift(img, shift=(shift_x, shift_y), mode="constant", cval=0)

        # Random rotation
        angle = np.random.uniform(-15, 15)
        rotated = rotate(shifted, angle=angle, reshape=False, mode="constant", cval=0)

        # Random scaling (zoom in/out)
        scale = np.random.uniform(0.9, 1.1)
        zoomed = zoom(rotated, zoom=scale)
        if zoomed.shape[0] > 28:
            zoomed = zoomed[:28, :28]
        else:
            pad_width = ((0, 28 - zoomed.shape[0]), (0, 28 - zoomed.shape[1]))
            zoomed = np.pad(zoomed, pad_width, mode="constant", constant_values=0)

        # Add small random noise
        noise = np.random.normal(0, 0.02, (28, 28))
        noised = zoomed + noise
        noised = np.clip(noised, 0, 1)

        X_augmented.append(noised.flatten())
        y_augmented.append(label)

    return np.array(X_augmented), np.array(y_augmented)


def objective(trial, X_train, y_train, X_test, y_test):
    """Objective function for Optuna hyperparameter tuning with MLflow logging."""
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    model.fit(X_train_aug, y_train_aug)
    train_accuracy = accuracy_score(y_train_aug, model.predict(X_train_aug))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))  # Test on clean data!

    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
            }
        )
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

    return test_accuracy


def train_model() -> Dict[str, str]:
    """Train RandomForest, optimize via Optuna, log everything with MLflow."""
    if mlflow.active_run() is not None:
        mlflow.end_run()
    try:
        db = get_db()
        X_train, y_train, X_test, y_test = load_data(db)

        if X_train.size == 0 or y_train.size == 0:
            raise HTTPException(status_code=404, detail="No valid training data found")
        if X_test.size == 0 or y_test.size == 0:
            raise HTTPException(status_code=404, detail="No valid test data found")

        logging.info("✅ Model training started")

        with mlflow.start_run(run_name="optuna_rf_tuning"):
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: objective(trial, X_train, y_train, X_test, y_test),
                n_trials=30,
            )

            best_params = study.best_params
            logging.info(f"Best hyperparameters: {best_params}")

            model = RandomForestClassifier(
                **best_params,
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )
            model.fit(X_train, y_train)

            train_accuracy = accuracy_score(y_train, model.predict(X_train))
            test_accuracy = accuracy_score(y_test, model.predict(X_test))

            mlflow.log_params(best_params)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            # mlflow.sklearn.log_model(model, artifact_path="model")
            # mlflow.log_artifact("src/models/model.pkl", artifact_path="model")
            logging.info(f"Best Train Accuracy: {train_accuracy}")
            logging.info(f"Best Test Accuracy: {test_accuracy}")

            # logging.info(f"Model saved in run {run.info.run_id}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, "model.pkl")
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model")

        model_dir = "src/models"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        logging.info("✅ Model training and logging successful")

        return {
            "status": "success",
            "message": "Model trained, tuned, and logged successfully",
            "code": 200,
        }

    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail="Model training unsuccessful")
