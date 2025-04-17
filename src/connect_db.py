import logging
import os
import pymongo
from fastapi import HTTPException
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load .env only if not running inside Docker
if not os.getenv("RUNNING_IN_DOCKER"):
    # Load .env from the parent directory
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

# Optional debug logs (consider removing them in production)
logging.debug(f"DEBUG: Loaded MONGO_URI = {os.getenv('MONGO_URI')}")
logging.debug(f"DEBUG: Loaded MONGO_DB_NAME = {os.getenv('MONGO_DB_NAME')}")

# 🔹 MongoDB Connection
def get_db():
    """Create a MongoDB client with proper handling."""
    try:
        # Use the environment variable MONGO_URI, fall back to localhost if not set
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "mnist_db")
        logging.info(f"Mongo URI: {mongo_uri}")

        logging.info("📥 Connecting to MongoDB...")
        client = pymongo.MongoClient(mongo_uri)
        db = client[mongo_db_name]

        logging.info("✅ Database connection successful")
        return db

    except Exception as e:
        logging.error(f"❌ MongoDB Connection Error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
