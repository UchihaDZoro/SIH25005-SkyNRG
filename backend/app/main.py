from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection
from app.api.routes import classification
import os

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Official Type Classification System as per Annex II",
    docs_url=f"{settings.API_V1_STR}/docs",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Events
@app.on_event("startup")
async def startup():
    await connect_to_mongo()
    print("=" * 60)
    print("✓ Animal Type Classification API Ready")
    print(f"✓ Using Official Format: 5 Sections, 20 Traits")
    print(f"✓ API Docs: http://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}/docs")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown():
    await close_mongo_connection()

# Routes
app.include_router(classification.router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Animal Type Classification API",
        "version": settings.VERSION,
        "format": "Official Type Evaluation Format (Annex II)",
        "sections": 5,
        "traits": 20,
        "docs": f"{settings.API_V1_STR}/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION
    }
