from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None
    
db = Database()

async def get_database():
    return db.client[settings.DATABASE_NAME]

async def connect_to_mongo():
    """Connect to MongoDB on startup"""
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    print(f"✓ Connected to MongoDB at {settings.MONGODB_URL}")

async def close_mongo_connection():
    """Close MongoDB connection on shutdown"""
    db.client.close()
    print("✓ Closed MongoDB connection")
