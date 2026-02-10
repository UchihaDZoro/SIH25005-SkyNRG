import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import MainLayout from './components/Layout/MainLayout'
import HomePage from './pages/HomePage'
import AboutPage from './pages/AboutPage'
import ClassifyPage from './pages/ClassifyPage'
import ContactPage from './pages/ContactPage'
import RGMPage from './pages/RGMPage'
import BreedsPage from './pages/BreedsPage'
import FAQPage from './pages/FAQPage'
import GalleryPage from './pages/GalleryPage'
import DocsPage from './pages/DocsPage'
import APIPage from './pages/APIPage'
import ArchivePage from './pages/ArchivePage'
import ClassificationResultsPage from './pages/ClassificationResultsPage'
import ScoringScalePage from './pages/ScoringScalePage'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<HomePage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="rgm" element={<RGMPage />} />
          <Route path="breeds" element={<BreedsPage />} />
          <Route path="classify" element={<ClassifyPage />} />
          <Route path="archive" element={<ArchivePage />} />
          <Route path="classification/:id" element={<ClassificationResultsPage />} />
          <Route path="scoring-scale" element={<ScoringScalePage />} />
          <Route path="contact" element={<ContactPage />} />
          <Route path="faq" element={<FAQPage />} />
          <Route path="gallery" element={<GalleryPage />} />
          <Route path="docs" element={<DocsPage />} />
          <Route path="api" element={<APIPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
