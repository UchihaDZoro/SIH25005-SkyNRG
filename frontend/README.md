# Animal Classification System - Frontend

A modern, responsive React application for the AI-based Animal Type Classification System. This application provides an intuitive interface for farmers and veterinarians to classify dairy cattle and buffaloes using the official government Type Evaluation Format (Annex II).

## 🚀 Technology Stack

- **Framework:** [React 19](https://react.dev/)
- **Build Tool:** [Vite](https://vitejs.dev/)
- **Styling:** [Tailwind CSS 4](https://tailwindcss.com/)
- **Routing:** [React Router v7](https://reactrouter.com/)
- **Animations:** 
  - [Framer Motion](https://www.framer.com/motion/) (Page transitions & UI animations)
  - [tsparticles](https://particles.js.org/) (Background effects)
  - [React Type Animation](https://www.npmjs.com/package/react-type-animation) (Typing effects)
- **HTTP Client:** [Axios](https://axios-http.com/)
- **Icons:** [React Icons](https://react-icons.github.io/react-icons/)
- **Data Handling:** 
  - [XLSX](https://docs.sheetjs.com/) (Excel export)
  - [File Saver](https://github.com/eligrey/FileSaver.js) (File downloads)

## ✨ Key Features

### 1. Interactive Classification Flow
- **Step-by-Step Wizard:** Guided process for entering animal details and uploading images.
- **Image Validation:** Ensures 3 standardized images (Left, Right, Udder/Rear) are uploaded.
- **Real-time Feedback:** Instant validation for required fields and file types.

### 2. Detailed Results Dashboard
- **Official Format:** Displays scores for all 20 official traits across 5 sections (Strength, Rump, Feet & Leg, Udder, General).
- **AI Analysis:** Shows AI-generated scores and measurements (in cm/degrees).
- **Milk Yield Prediction:** Estimates daily and lactation milk yield based on body measurements.
- **Animal Details:** Comprehensive view of animal identity, breed, and owner information.

### 3. Archive & Management
- **Search & Filter:** Filter classifications by animal type, breed, village, or grade.
- **Export:** Download detailed reports in Excel format.
- **Management:** Delete old or incorrect records.

## 🛠️ Installation & Setup

1. **Prerequisites**
   - Node.js (v18 or higher)
   - npm or yarn

2. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

3. **Run Development Server**
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173`.

4. **Build for Production**
   ```bash
   npm run build
   ```

## 📂 Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/           # Main page views
│   │   ├── HomePage.jsx
│   │   ├── ClassifyPage.jsx
│   │   ├── ClassificationResultsPage.jsx
│   │   └── ArchivePage.jsx
│   ├── services/        # API integration (axios)
│   ├── App.jsx          # Main application component
│   └── main.jsx         # Entry point
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── eslint.config.js     # Linting configuration
```

## 🎨 Design System

- **Color Palette:**
  - **Primary:** Black & White (Modern, minimal base)
  - **Accents:** Orange (Actions), Green (Success/Verified), Blue (Information)
  - **Gradients:** Subtle teal/blue gradients for headers.

- **Typography:** Clean sans-serif fonts for readability.

- **Components:**
  - **Cards:** White cards with soft shadows and rounded corners.
  - **Buttons:** High-contrast buttons with hover effects.
  - **Inputs:** Clean, focus-ring enhanced form fields.

## 🔗 API Integration

The frontend communicates with the FastAPI backend via `src/services/api.js`.
- **Base URL:** `http://127.0.0.1:8000/api/v1`
- **Endpoints:**
  - `POST /classification/create`: Initialize classification
  - `POST /classification/{id}/upload-images`: Upload images
  - `POST /classification/{id}/process`: Trigger AI analysis
  - `GET /classification/{id}/results`: Fetch results
  - `GET /classification/archive`: Fetch history

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
