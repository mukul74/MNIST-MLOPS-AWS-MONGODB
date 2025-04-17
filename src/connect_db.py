import logging
import os
import pymongo
from fastapi import HTTPException
from dotenv import load_dotenv
from pathlib import Path

# Setup logging level
logging.basicConfig(level=logging.INFO)

# ✅ Load environment variables if not in Docker
if not os.getenv("RUNNING_IN_DOCKER"):
    print("🧪 Running locally. Loading .env...")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
else:
    print("🐳 Running in Docker. Skipping .env load.")

# 🔹 MongoDB Connection
def get_db():
    """Create a MongoDB client with proper handling."""
    try:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        mongo_db_name = os.getenv("MONGO_DB_NAME", "mnist_db")

        logging.info(f"Mongo URI: {mongo_uri}")
        logging.info("📥 Connecting to MongoDB...")
        
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[mongo_db_name]

        # Trigger a call to check the connection
        client.server_info()

        logging.info("✅ Database connection successful")
        return db

    except Exception as e:
        logging.error(f"❌ MongoDB Connection Error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
