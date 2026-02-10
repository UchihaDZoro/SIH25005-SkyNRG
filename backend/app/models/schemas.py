from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AnimalType(str, Enum):
    CATTLE = "cattle"
    BUFFALO = "buffalo"

class AnimalInfo(BaseModel):
    tagNumber: Optional[str] = None
    animalType: Optional[AnimalType] = AnimalType.CATTLE
    breed: Optional[str] = "Unknown"
    dateOfBirth: Optional[str] = ""
    lactationNumber: Optional[int] = Field(default=1, ge=1, le=10)
    dateOfCalving: Optional[str] = ""
    village: Optional[str] = "Not specified"
    farmerName: Optional[str] = "Unknown"
    farmerContact: Optional[str] = None

class ClassificationCreate(BaseModel):
    animalInfo: Optional[AnimalInfo] = None

class ImageInfo(BaseModel):
    filename: str
    url: str
    angle: str
    status: str

class TraitScore(BaseModel):
    trait: str
    score: int = Field(ge=1, le=9)
    measurement: Optional[float] = None

class OfficialFormatResponse(BaseModel):
    villageName: str
    farmerName: str
    animalTagNo: str
    dateOfBirth: str
    lactationNo: int
    dateOfCalving: str
    classificationDate: str
    classifiedBy: str
    sections: Dict[str, List[TraitScore]]

class ClassificationResult(BaseModel):
    id: str
    animalInfo: AnimalInfo
    officialFormat: OfficialFormatResponse
    categoryScores: Dict[str, float]
    overallScore: float
    grade: str
    totalTraits: int
    confidenceLevel: int
    createdAt: datetime
    status: str
