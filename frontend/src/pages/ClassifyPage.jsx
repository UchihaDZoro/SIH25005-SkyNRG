import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FiUploadCloud, FiX, FiAlertCircle, FiImage } from 'react-icons/fi';
import axios from 'axios';

const ClassifyPage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for 5 separate images
  const [images, setImages] = useState({
    rear: null,
    side: null,
    top: null,
    udder: null,
    side_udder: null
  });

  const [previews, setPreviews] = useState({
    rear: null,
    side: null,
    top: null,
    udder: null,
    side_udder: null
  });

  // View definitions with labels
  const viewDefinitions = [
    { key: 'rear', label: 'Rear View', color: 'blue' },
    { key: 'side', label: 'Side View', color: 'green' },
    { key: 'top', label: 'Top View', color: 'purple' },
    { key: 'udder', label: 'Udder View', color: 'orange' },
    { key: 'side_udder', label: 'Side Udder View', color: 'teal' }
  ];

  const handleImageSelect = (viewKey, e) => {
    const file = e.target.files[0];
    
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      setError(`Please select a valid image file for ${viewKey}`);
      return;
    }
    
    setImages(prev => ({ ...prev, [viewKey]: file }));
    setPreviews(prev => ({ ...prev, [viewKey]: URL.createObjectURL(file) }));
    setError(null);
  };

  const removeImage = (viewKey) => {
    setImages(prev => ({ ...prev, [viewKey]: null }));
    setPreviews(prev => ({ ...prev, [viewKey]: null }));
  };

  const validateForm = () => {
    const missingViews = viewDefinitions
      .filter(view => !images[view.key])
      .map(view => view.label);
    
    if (missingViews.length > 0) {
      setError(`Please upload images for: ${missingViews.join(', ')}`);
      return false;
    }
    
    setError(null);
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setLoading(true);
    setError(null);

    try {
      const createResponse = await axios.post('http://127.0.0.1:8000/api/v1/classification/create', {});
      if (!createResponse.data.success) throw new Error("Failed to create record");
      
      const classificationId = createResponse.data.data.id;

      // Prepare images in the correct order
      const imageFormData = new FormData();
      viewDefinitions.forEach(view => {
        imageFormData.append('images', images[view.key]);
      });

      const uploadResponse = await axios.post(
        `http://127.0.0.1:8000/api/v1/classification/${classificationId}/upload-images`,
        imageFormData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      if (!uploadResponse.data.success) throw new Error("Failed to upload images");

      const processResponse = await axios.post(
        `http://127.0.0.1:8000/api/v1/classification/${classificationId}/process`
      );

      if (processResponse.data.success) {
        navigate(`/classification/${classificationId}`);
      } else {
        throw new Error("Processing failed");
      }

    } catch (err) {
      console.error("Classification error:", err);
      let errorMessage = "An unexpected error occurred.";
      
      if (err.response?.data?.detail) {
        const detail = err.response.data.detail;
        if (Array.isArray(detail)) {
          errorMessage = detail.map(e => `${e.loc.join('.')}: ${e.msg}`).join(', ');
        } else if (typeof detail === 'object') {
          errorMessage = JSON.stringify(detail);
        } else {
          errorMessage = detail;
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getColorClasses = (color, type = 'border') => {
    const colorMap = {
      blue: {
        border: 'border-blue-500',
        bg: 'bg-blue-50',
        text: 'text-blue-600',
        hover: 'hover:border-blue-500'
      },
      green: {
        border: 'border-green-500',
        bg: 'bg-green-50',
        text: 'text-green-600',
        hover: 'hover:border-green-500'
      },
      purple: {
        border: 'border-purple-500',
        bg: 'bg-purple-50',
        text: 'text-purple-600',
        hover: 'hover:border-purple-500'
      },
      orange: {
        border: 'border-orange-500',
        bg: 'bg-orange-50',
        text: 'text-orange-600',
        hover: 'hover:border-orange-500'
      },
      teal: {
        border: 'border-teal-500',
        bg: 'bg-teal-50',
        text: 'text-teal-600',
        hover: 'hover:border-teal-500'
      }
    };
    return colorMap[color]?.[type] || '';
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-20">
      {/* Header */}
      <section className="bg-gradient-to-r from-slate-900 to-slate-800 text-white py-16 border-b-4 border-orange-500">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center">
            <p className="text-xs font-bold tracking-widest uppercase text-gray-400 mb-4">Official Type Evaluation System</p>
            <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">Classify Livestock</h1>
            <p className="text-lg text-gray-300 leading-relaxed max-w-3xl mx-auto font-medium">
              Upload 5 standardized images of specific cattle views for comprehensive type score analysis following the official Annex II format with 20 traits across 5 sections.
            </p>
          </div>
        </div>
      </section>

      {/* Form Section */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <form onSubmit={handleSubmit} className="space-y-8">
            
            {/* Upload Instructions */}
            <div className="bg-white border-2 border-gray-200 p-6">
              <div className="flex items-start gap-4">
                <FiImage className="w-6 h-6 text-orange-500 flex-shrink-0 mt-1" />
                <div>
                  <h3 className="text-sm font-bold uppercase tracking-wide mb-2 text-gray-700">Upload Instructions</h3>
                  <p className="text-sm text-gray-600 font-medium leading-relaxed">
                    Please upload one image for each of the 5 specific cattle views below. Each view is essential for accurate type evaluation.
                  </p>
                </div>
              </div>
            </div>

            {/* 5 Upload Boxes */}
            <div className="bg-white border-2 border-gray-200 overflow-hidden">
              <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
                <h2 className="text-xl font-bold uppercase tracking-wide">Upload Images for Each View</h2>
                <p className="text-sm text-gray-300 mt-2 font-medium">
                  {Object.values(images).filter(Boolean).length} of 5 images uploaded
                </p>
              </div>

              <div className="p-8">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {viewDefinitions.map((view) => (
                    <div key={view.key} className="flex flex-col">
                      {/* View Label */}
                      <div className={`px-4 py-2 ${getColorClasses(view.color, 'bg')} border-2 ${getColorClasses(view.color, 'border')} mb-2`}>
                        <h3 className={`text-sm font-bold uppercase tracking-wide ${getColorClasses(view.color, 'text')}`}>
                          {view.label}
                        </h3>
                      </div>

                      {/* Upload Box */}
                      {!previews[view.key] ? (
                        <div className="relative flex-1">
                          <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleImageSelect(view.key, e)}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                          />
                          <div className={`border-4 border-dashed border-gray-300 ${getColorClasses(view.color, 'hover')} transition-all bg-white p-8 text-center cursor-pointer h-full flex flex-col items-center justify-center min-h-[200px]`}>
                            <div className={`w-16 h-16 ${getColorClasses(view.color, 'bg')} flex items-center justify-center mb-4 border-2 ${getColorClasses(view.color, 'border')}`}>
                              <FiUploadCloud className={`w-8 h-8 ${getColorClasses(view.color, 'text')}`} />
                            </div>
                            <p className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-2">
                              Click to Upload
                            </p>
                            <p className="text-xs text-gray-500 font-medium">JPG, PNG, JPEG</p>
                          </div>
                        </div>
                      ) : (
                        <div className="relative group flex-1">
                          <div className={`border-2 ${getColorClasses(view.color, 'border')} bg-gray-200 overflow-hidden relative`}>
                            <img
                              src={previews[view.key]}
                              alt={view.label}
                              className="w-full h-48 object-cover"
                            />
                            <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
                              <button
                                type="button"
                                onClick={() => removeImage(view.key)}
                                className="w-12 h-12 bg-red-500 text-white flex items-center justify-center hover:bg-red-600 transition-colors border-2 border-red-700"
                              >
                                <FiX className="w-6 h-6" />
                              </button>
                              <label className="w-12 h-12 bg-blue-500 text-white flex items-center justify-center hover:bg-blue-600 transition-colors border-2 border-blue-700 cursor-pointer">
                                <FiUploadCloud className="w-6 h-6" />
                                <input
                                  type="file"
                                  accept="image/*"
                                  onChange={(e) => handleImageSelect(view.key, e)}
                                  className="hidden"
                                />
                              </label>
                            </div>
                          </div>
                          <div className={`mt-2 px-3 py-1 ${getColorClasses(view.color, 'bg')} border ${getColorClasses(view.color, 'border')} text-center`}>
                            <span className={`text-xs font-bold uppercase ${getColorClasses(view.color, 'text')}`}>✓ Uploaded</span>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-6 flex items-start gap-3">
                <FiAlertCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-bold text-red-900 mb-1 uppercase text-sm tracking-wide">Error</h3>
                  <p className="text-sm text-red-700 font-medium">{error}</p>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <div className="flex justify-center pt-4">
              <button
                type="submit"
                disabled={loading}
                className={`px-16 py-5 bg-orange-500 text-white text-base font-bold tracking-widest uppercase border-b-4 border-orange-700 ${
                  loading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-orange-600'
                } transition-colors`}
              >
                {loading ? (
                  <span className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white animate-spin"></div>
                    Processing...
                  </span>
                ) : (
                  'Start Classification'
                )}
              </button>
            </div>
          </form>
        </div>
      </section>

      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-white/95 backdrop-blur-md z-50 flex flex-col items-center justify-center">
          <div className="w-20 h-20 border-4 border-gray-300 border-t-orange-500 animate-spin mb-6"></div>
          <h3 className="text-2xl font-bold text-black tracking-wide mb-2 uppercase">Processing Classification</h3>
          <p className="text-gray-600 text-sm uppercase tracking-wider font-semibold">Analyzing images and computing trait scores...</p>
        </div>
      )}
    </div>
  );
};

export default ClassifyPage;
