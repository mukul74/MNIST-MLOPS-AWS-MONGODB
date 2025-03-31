import logging
from fastapi import HTTPException
from sklearn.datasets import fetch_openml
from src.connect_db import get_db

# Configure Logging
logging.basicConfig(level=logging.INFO)

def store_mnist_model():
    """Fetches the MNIST dataset and stores it in MongoDB if not already present."""
    try:
        logging.info("📥 Connecting to MongoDB...")

        # 🔹 Get MongoDB
        db = get_db()
        train_collection = db["train_data"]
        test_collection = db["test_data"]

        logging.info("✅ Connected to MongoDB")

        # 🔹 Check if collections exist and contain data
        if train_collection.estimated_document_count() > 0 and test_collection.estimated_document_count() > 0:
            logging.info("⚠️ MNIST data already exists in MongoDB. Skipping insertion.")
            return {"status": "exists", "message": "MNIST Data already exists in MongoDB", "code" : 200}

        # 🔹 Fetch MNIST dataset
        logging.info("📥 Fetching MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1, parser="pandas")
        X = mnist.data.to_numpy()
        y = mnist.target.to_numpy().astype(int)

        # 🔹 Split into train & test
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        # 🔹 Convert data to MongoDB format
        logging.info("📦 Preparing data for MongoDB storage...")
        train_data = [{"features": x.tolist(), "label": int(y)} for x, y in zip(X_train, y_train)]
        test_data = [{"features": x.tolist(), "label": int(y)} for x, y in zip(X_test, y_test)]

        # 🔹 Store in MongoDB (Bulk Insert)
        train_collection.insert_many(train_data)
        test_collection.insert_many(test_data)

        logging.info("✅ MNIST Data Successfully Stored in MongoDB")
        return {"status": "success", "message": "MNIST Data Stored in MongoDB", "code":200}

    except Exception as e:
        logging.error(f"❌ Failed to store MNIST data: {e}")
        raise HTTPException(status_code=500, detail="Failed to store MNIST data")
