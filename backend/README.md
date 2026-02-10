# Animal Type Classification Backend

Official Type Classification System as per Annex II - 5 Sections, 20 Traits

## Features

✅ **Official Government Format**: Exact implementation of Type Evaluation Format (Annex II)  
✅ **20 Traits**: Comprehensive scoring across 5 sections  
✅ **MongoDB Storage**: Robust database for classification records  
✅ **Gemini AI Integration**: Easily swappable with custom models  
✅ **Mock Data Fallback**: Works without AI for development/testing  
✅ **Complete API**: RESTful endpoints for full workflow  

## Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── core/
│   │   ├── config.py          # Settings & configuration
│   │   └── database.py        # MongoDB connection
│   ├── models/
│   │   ├── schemas.py         # Pydantic models
│   │   └── trait_definitions.py  # Official 20 traits
│   ├── services/
│   │   └── ai_service.py      # Gemini AI integration
│   └── api/routes/
│       └── classification.py  # API endpoints
├── uploads/                    # Image storage
├── requirements.txt           # Dependencies
└── .env                       # Environment variables
```

## Official Traits (20 Total)

### Section 1: STRENGTH (5 traits)
1. Stature
2. Heart Girth
3. Body Length
4. Body Depth
5. Angularity

### Section 2: RUMP (2 traits)
6. Rump Angle
7. Rump Width

### Section 3: FEET AND LEG (3 traits)
8. Rear Legs Set
9. Rear Legs Rear View
10. Foot Angle

### Section 4: UDDER (9 traits)
11. Fore Udder Attachment
12. Rear Udder Height
13. Central Ligament
14. Udder Depth
15. Front Teat Placement
16. Teat Length
17. Rear Teat Placement
18. Rear udder width
19. Teat thickness

### Section 5: GENERAL (1 trait)
20. Body condition score

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment**:
Edit `.env` file:
```env
GEMINI_API_KEY=your_actual_api_key_here
MONGODB_URL=mongodb://localhost:27017
```

3. **Start MongoDB** (if not running):
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

4. **Run Server**:
```bash
uvicorn app.main:app --reload --port 8000
```

API will be available at: http://localhost:8000

## API Documentation

Interactive docs: http://localhost:8000/api/v1/docs

### Workflow

#### 1. Create Classification
```http
POST /api/v1/classification/create
Content-Type: application/json

{
  "animalInfo": {
    "tagNumber": "ABC123",
    "animalType": "cattle",
    "breed": "Gir",
    "dateOfBirth": "2020-01-15",
    "lactationNumber": 2,
    "dateOfCalving": "2024-06-01",
    "village": "Sample Village",
    "farmerName": "John Doe"
  }
}
```

**Response**: Returns `id` for classification

#### 2. Upload Images (6 required)
```http
POST /api/v1/classification/{id}/upload-images
Content-Type: multipart/form-data

images: [left_side.jpg, right_side.jpg, rear.jpg, front.jpg, udder.jpg, top.jpg]
```

Angles must be in order:
1. left_side
2. right_side
3. rear
4. front
5. udder_closeup
6. top_view

#### 3. Process Classification
```http
POST /api/v1/classification/{id}/process
```

AI processes images and generates official format results

#### 4. Get Results
```http
GET /api/v1/classification/{id}/results
```

**Response**:
```json
{
  "success": true,
  "data": {
    "officialFormat": {
      "villageName": "Sample Village",
      "farmerName": "John Doe",
      "animalTagNo": "ABC123",
      "classificationDate": "2025-12-02",
      "sections": {
        "Strength": [
          {"trait": "Stature", "score": 7, "measurement": 138},
          {"trait": "Heart Girth", "score": 6, "measurement": 195},
          ...
        ],
        "Rump": [...],
        "Feet and Leg": [...],
        "Udder": [...],
        "General": [...]
      }
    },
    "categoryScores": {
      "Strength": 6.4,
      "Rump": 6.5,
      ...
    },
    "overallScore": 6.5,
    "grade": "Good",
    "totalTraits": 20
  }
}
```

#### 5. Check Status
```http
GET /api/v1/classification/{id}/status
```

#### 6. List All
```http
GET /api/v1/classification/list?limit=10&skip=0
```

## AI Integration

### Using Gemini AI (Default)

Get API key from: https://makersuite.google.com/app/apikey

Add to `.env`:
```env
GEMINI_API_KEY=your_key_here
```

### Using Custom Model

Replace in `app/services/ai_service.py`:

```python
class AIService:
    async def classify_animal(self, image_paths, animal_info):
        # Your custom model logic here
        results = your_model.predict(image_paths)
        return self._format_results(results, animal_info)
```

### Mock Data (No AI)

If `GEMINI_API_KEY` is not set, system automatically uses realistic mock data following the exact official format.

## Development

**Hot Reload**:
```bash
uvicorn app.main:app --reload --port 8000
```

**Check Health**:
```bash
curl http://localhost:8000/health
```

**View Logs**:
Server logs show MongoDB connection, API ready status, and processing steps.

## Production Notes

1. **Security**: Add authentication/authorization
2. **Rate Limiting**: Implement for public APIs
3. **Image Validation**: Add virus scanning
4. **Storage**: Use cloud storage (S3, GCS) instead of local
5. **Database**: Use MongoDB Atlas or managed instance
6. **Monitoring**: Add logging and error tracking

## Troubleshooting

**MongoDB Connection Failed**:
- Ensure MongoDB is running on port 27017
- Check `MONGODB_URL` in `.env`

**Gemini API Error**:
- Verify `GEMINI_API_KEY` is correct
- System will fall back to mock data automatically

**Import Errors**:
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version >= 3.9

## License

MIT License - Educational/Government Use
