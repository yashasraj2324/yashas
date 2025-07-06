from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import List, Dict

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "pii_masker"
COLLECTION_NAME = "uploads"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

async def save_upload_info(filename: str, pii: List[Dict], timestamp: datetime = None):
    if timestamp is None:
        timestamp = datetime.utcnow()
    doc = {
        "filename": filename,
        "pii": pii,
        "timestamp": timestamp
    }
    await collection.insert_one(doc) 