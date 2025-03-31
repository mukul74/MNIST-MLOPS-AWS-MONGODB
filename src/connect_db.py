import os
import pymongo
from fastapi import HTTPException
import logging

# 🔹 MongoDB Connection
def get_db():
    """Create a MongoDB client with proper handling."""
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        client = pymongo.MongoClient(mongo_uri)
        db = client["mnist_db"]
        logging.info("Database connection succesfull")
        return db
    except Exception as e:
        logging.error(f"MongoDB Connection Error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")