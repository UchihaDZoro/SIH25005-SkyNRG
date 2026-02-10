from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
from app.models.schemas import *
from app.services.ai_service import ai_service
from app.core.config import settings
from app.core.database import get_database
import aiofiles
import os
from datetime import datetime
from bson import ObjectId

router = APIRouter(prefix="/classification", tags=["Classification"])

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@router.post("/create", status_code=201)
async def create_classification(data: ClassificationCreate):
    """Step 1: Create classification record"""
    
    db = await get_database()
    
    # Handle missing or empty animalInfo
    if data.animalInfo is None:
        from app.models.schemas import AnimalInfo
        data.animalInfo = AnimalInfo()
    
    # Auto-generate tag number if not provided
    if not data.animalInfo.tagNumber:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        data.animalInfo.tagNumber = f"AUTO-{timestamp}"
    
    classification = {
        "animalInfo": data.animalInfo.dict(),
        "status": "created",
        "images": [],
        "results": None,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }
    
    result = await db.classifications.insert_one(classification)
    
    return {
        "success": True,
        "message": "Classification created",
        "data": {
            "id": str(result.inserted_id),
            "status": "created"
        }
    }

@router.post("/{classification_id}/upload-images")
async def upload_images(
    classification_id: str,
    images: List[UploadFile] = File(...),
):
    """Step 2: Upload 5 images for view classification"""
    
    if len(images) != 5:
        raise HTTPException(400, "Exactly 5 images required")
    
    db = await get_database()
    
    try:
        classification = await db.classifications.find_one({"_id": ObjectId(classification_id)})
    except Exception as e:
        raise HTTPException(400, f"Invalid classification ID: {str(e)}")
    
    if not classification:
        raise HTTPException(404, "Classification not found")
    
    # Specific labels for each cattle view
    angles = ["rear", "side", "top", "udder", "side_udder"]
    
    uploaded_files = []
    
    for idx, (image, angle) in enumerate(zip(images, angles)):
        if not image.content_type.startswith('image/'):
            raise HTTPException(400, f"File {image.filename} is not an image")
        
        ext = os.path.splitext(image.filename)[1]
        filename = f"{classification_id}_{angle}{ext}"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            content = await image.read()
            await f.write(content)
        
        uploaded_files.append({
            "filename": filename,
            "url": f"/uploads/{filename}",
            "angle": angle,
            "status": "uploaded"
        })
    
    await db.classifications.update_one(
        {"_id": ObjectId(classification_id)},
        {
            "$set": {
                "images": uploaded_files,
                "status": "images_uploaded",
                "updatedAt": datetime.utcnow()
            }
        }
    )
    
    return {
        "success": True,
        "message": "Images uploaded successfully",
        "data": uploaded_files
    }

@router.post("/{classification_id}/process")
async def process_classification(classification_id: str):
    """Step 3: Process with AI following official format"""
    
    db = await get_database()
    
    try:
        classification = await db.classifications.find_one({"_id": ObjectId(classification_id)})
    except Exception as e:
        raise HTTPException(400, f"Invalid classification ID: {str(e)}")
    
    if not classification:
        raise HTTPException(404, "Classification not found")
    
    if not classification.get('images'):
        raise HTTPException(400, "No images uploaded")
    
    await db.classifications.update_one(
        {"_id": ObjectId(classification_id)},
        {"$set": {"status": "processing", "updatedAt": datetime.utcnow()}}
    )
    
    try:
        image_paths = [
            os.path.join(settings.UPLOAD_DIR, img['filename'])
            for img in classification['images']
        ]
        
        # AI Classification with official format
        results = await ai_service.classify_animal(
            image_paths=image_paths,
            animal_info=classification['animalInfo']
        )
        
        await db.classifications.update_one(
            {"_id": ObjectId(classification_id)},
            {
                "$set": {
                    "results": results,
                    "status": "completed",
                    "updatedAt": datetime.utcnow()
                }
            }
        )
        
        return {
            "success": True,
            "message": "Classification completed using official Type Evaluation Format",
            "data": {
                "id": classification_id,
                "status": "completed",
                "totalTraits": 20
            }
        }
        
    except Exception as e:
        await db.classifications.update_one(
            {"_id": ObjectId(classification_id)},
            {"$set": {"status": "failed", "error": str(e), "updatedAt": datetime.utcnow()}}
        )
        raise HTTPException(500, f"Processing failed: {str(e)}")

@router.get("/{classification_id}/results")
async def get_results(classification_id: str):
    """Step 4: Get official format results"""
    
    db = await get_database()
    
    try:
        classification = await db.classifications.find_one({"_id": ObjectId(classification_id)})
    except Exception as e:
        raise HTTPException(400, f"Invalid classification ID: {str(e)}")
    
    if not classification:
        raise HTTPException(404, "Classification not found")
    
    if classification['status'] != 'completed':
        return {
            "success": True,
            "data": {
                "id": classification_id,
                "status": classification['status']
            }
        }
    
    results = classification['results']
    results['id'] = classification_id
    results['createdAt'] = classification['createdAt']
    results['status'] = 'completed'
    
    return {
        "success": True,
        "message": "Results in official Type Evaluation Format",
        "data": results
    }

@router.get("/{classification_id}/status")
async def get_status(classification_id: str):
    """Get processing status"""
    
    db = await get_database()
    
    try:
        classification = await db.classifications.find_one({"_id": ObjectId(classification_id)})
    except Exception as e:
        raise HTTPException(400, f"Invalid classification ID: {str(e)}")
    
    if not classification:
        raise HTTPException(404, "Classification not found")
    
    return {
        "success": True,
        "data": {
            "id": classification_id,
            "status": classification['status'],
            "updatedAt": classification['updatedAt']
        }
    }

@router.get("/list")
async def list_classifications(limit: int = 10, skip: int = 0):
    """List all classifications"""
    
    db = await get_database()
    
    cursor = db.classifications.find().sort("createdAt", -1).skip(skip).limit(limit)
    classifications = await cursor.to_list(length=limit)
    
    # Convert ObjectId to string
    for c in classifications:
        c['id'] = str(c['_id'])
        del c['_id']
    
    return {
        "success": True,
        "data": classifications,
        "count": len(classifications)
    }

@router.get("/archive")
async def get_archive(
    skip: int = 0,
    limit: int = 20,
    animal_type: Optional[str] = None,
    breed: Optional[str] = None,
    village: Optional[str] = None,
    grade: Optional[str] = None,
    search: Optional[str] = None
):
    """Get all completed classifications with filters and pagination for archive page"""
    
    db = await get_database()
    
    # Build filter query - only completed classifications
    query = {"status": "completed"}
    
    # Apply filters if provided
    if animal_type:
        query["animalInfo.animalType"] = animal_type
    if breed:
        query["animalInfo.breed"] = {"$regex": breed, "$options": "i"}  # Case-insensitive partial match
    if village:
        query["animalInfo.village"] = {"$regex": village, "$options": "i"}
    if grade:
        query["results.grade"] = grade
    
    # Search across tag number, breed, village, farmer name
    if search:
        query["$or"] = [
            {"animalInfo.tagNumber": {"$regex": search, "$options": "i"}},
            {"animalInfo.breed": {"$regex": search, "$options": "i"}},
            {"animalInfo.village": {"$regex": search, "$options": "i"}},
            {"animalInfo.farmerName": {"$regex": search, "$options": "i"}}
        ]
    
    # Get total count for pagination
    total = await db.classifications.count_documents(query)
    
    # Get paginated results
    classifications = await db.classifications.find(query)\
        .sort("createdAt", -1)\
        .skip(skip)\
        .limit(limit)\
        .to_list(length=limit)
    
    # Format results for archive display
    results = []
    for c in classifications:
        results.append({
            "id": str(c["_id"]),
            "tagNumber": c["animalInfo"]["tagNumber"],
            "animalType": c["animalInfo"]["animalType"],
            "breed": c["animalInfo"]["breed"],
            "village": c["animalInfo"]["village"],
            "farmerName": c["animalInfo"]["farmerName"],
            "overallScore": c["results"]["overallScore"],
            "grade": c["results"]["grade"],
            "createdAt": c["createdAt"].isoformat(),
            "confidenceLevel": c["results"].get("confidenceLevel", "High")
        })
    
    return {
        "success": True,
        "data": {
            "results": results,
            "total": total,
            "page": skip // limit + 1 if limit > 0 else 1,
            "totalPages": (total + limit - 1) // limit if limit > 0 else 1,
            "limit": limit,
            "skip": skip
        }
    }

@router.delete("/{classification_id}")
async def delete_classification(classification_id: str):
    """Delete a classification by ID"""
    
    db = await get_database()
    
    try:
        # Check if classification exists
        classification = await db.classifications.find_one({"_id": ObjectId(classification_id)})
    except Exception as e:
        raise HTTPException(400, f"Invalid classification ID: {str(e)}")
    
    if not classification:
        raise HTTPException(404, "Classification not found")
    
    # Delete the classification
    result = await db.classifications.delete_one({"_id": ObjectId(classification_id)})
    
    if result.deleted_count == 0:
        raise HTTPException(500, "Failed to delete classification")
    
    return {
        "success": True,
        "message": "Classification deleted successfully",
        "data": {
            "id": classification_id
        }
    }
