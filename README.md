# 🐄 Animal Classification System

> AI-powered cattle type classification system with official government format compliance (Annex II - 20 Traits)

## 🌟 Features

- **🎯 Official Government Format**: Exact implementation of Type Evaluation Format (Annex II)
- **📊 20 Trait Analysis**: Comprehensive scoring across 5 sections
- **🤖 AI-Powered**: Gemini AI integration for intelligent classification
- **🌍 Multi-Language Support**: 22 Indian languages (Hindi, Tamil, Telugu, Bengali, etc.)
- **📱 Responsive Design**: Modern, mobile-friendly interface
- **💾 MongoDB Storage**: Robust database for classification records
- **📁 Archive & Export**: Excel export functionality for reports

## 🏗️ Tech Stack

### Frontend
- **React 18** + **Vite** - Modern, fast development
- **React Router v6** - Client-side routing
- **Tailwind CSS** - Utility-first styling
- **i18next** - Internationalization (22 languages)
- **Framer Motion** - Smooth animations
- **Axios** - HTTP client

### Backend
- **Python 3.9+** - Core language
- **FastAPI** - Modern, high-performance API framework
- **MongoDB** - NoSQL database
- **Gemini AI** - Google's generative AI model
- **Uvicorn** - ASGI server

## 📁 Project Structure

```
animal-classification-system/
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── i18n/         # 22 language translations
│   │   └── main.jsx      # Entry point
│   ├── .env.example      # Environment template
│   └── package.json
│
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── core/         # Configuration
│   │   ├── models/       # Data models & schemas
│   │   └── services/     # Business logic & AI
│   ├── .env.example      # Environment template
│   └── requirements.txt
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **MongoDB** (local or Atlas)
- **Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/animal-classification-system.git
cd animal-classification-system
```

### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
copy .env.example .env
# Edit .env with your MongoDB URL and Gemini API key

# Start server
python -m uvicorn app.main:app --reload
```

Backend will run at: **http://localhost:8000**

### 3️⃣ Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Setup environment variables
copy .env.example .env
# Edit .env if needed (default points to localhost:8000)

# Start development server
npm run dev
```

Frontend will run at: **http://localhost:5173**

## 🔐 Environment Variables

### Backend (.env)

```env
# MongoDB
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
DATABASE_NAME=animal_classification

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here

# Server
PORT=8000
DEBUG=True
```

### Frontend (.env)

```env
VITE_API_BASE_URL=http://localhost:8000
```

## 📖 Usage

1. **Navigate** to http://localhost:5173
2. **Select language** from the language switcher (22 languages available)
3. **Go to Classify** page
4. **Fill animal information** (tag number, breed, DOB, etc.)
5. **Upload 6 required images**:
   - Left side
   - Right side
   - Rear view
   - Front view
   - Udder closeup
   - Top view
6. **Submit for classification**
7. **View results** with official 20-trait scoring
8. **Export to Excel** or save to archive

## 🎯 Official 20 Traits Coverage

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
18. Rear Udder Width
19. Teat Thickness

### Section 5: GENERAL (1 trait)
20. Body Condition Score

## 🌍 Supported Languages

**22 Indian Languages**: Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Bengali, Punjabi, Odia, Assamese, Urdu, Kashmiri, Konkani, Nepali, Bodo, Dogri, Maithili, Manipuri, Santali, Sindhi

## 📚 API Documentation

Interactive API docs available at: **http://localhost:8000/docs**

### Key Endpoints

- `POST /api/v1/classification/create` - Create new classification
- `POST /api/v1/classification/{id}/upload-images` - Upload 6 images
- `POST /api/v1/classification/{id}/process` - Process with AI
- `GET /api/v1/classification/{id}/results` - Get results
- `GET /api/v1/classification/list` - List all classifications

## 🛠️ Development

### Frontend Commands

```bash
npm run dev          # Development server
npm run build        # Production build
npm run preview      # Preview production build
```

### Backend Commands

```bash
uvicorn app.main:app --reload   # Development server with hot reload
pyt0hon -m pytest                # Run tests (if available)
```

## 👥 Contributors

Nancy Srivastava | Pathan Gulamgaush | Swasti Mishra | Rabia Basriya | Sriyukthika | Kunal Verma

## 🙏 Acknowledgments

- Official Government Type Evaluation Format (Annex II)
- Google Gemini AI
- FastAPI & React communities

---

**Made with ❤️ for Smart India Hackathon 2025**
