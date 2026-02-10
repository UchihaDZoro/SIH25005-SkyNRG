import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FiCheckCircle, FiAlertCircle, FiAward, FiActivity } from 'react-icons/fi';
import axios from 'axios';

const ClassificationResultsPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchResults();
  }, [id]);

  const fetchResults = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`http://127.0.0.1:8000/api/v1/classification/${id}/results`);
      console.log('Results received:', response.data);

      if (response.data.success) {
        setResults(response.data.data);
      } else {
        setError('Failed to fetch results');
      }
    } catch (err) {
      console.error('Results fetch error:', err);
      setError(err.response?.data?.detail || 'Failed to load classification results');
    } finally {
      setLoading(false);
    }
  };

  const getGradeColor = (grade) => {
    const colors = {
      'Excellent': 'text-green-600',
      'Good': 'text-blue-600',
      'Fair': 'text-orange-600',
      'Poor': 'text-red-600'
    };
    return colors[grade] || 'text-gray-600';
  };

  const getGradeBgColor = (grade) => {
    const colors = {
      'Excellent': 'bg-green-50 border-green-500',
      'Good': 'bg-blue-50 border-blue-500',
      'Fair': 'bg-orange-50 border-orange-500',
      'Poor': 'bg-red-50 border-red-500'
    };
    return colors[grade] || 'bg-gray-50 border-gray-500';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center pt-20">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-gray-300 border-t-orange-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-lg font-bold uppercase text-gray-900">Loading Results...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 pt-20 px-4">
        <div className="max-w-2xl mx-auto mt-20">
          <div className="bg-red-50 border-l-4 border-red-500 p-6">
            <div className="flex items-start">
              <FiAlertCircle className="w-6 h-6 text-red-500 mt-1 mr-3" />
              <div>
                <h3 className="font-bold text-red-900 mb-2 uppercase">Error Loading Results</h3>
                <p className="text-red-700">{error}</p>
              </div>
            </div>
            <div className="mt-6 flex gap-4">
              <button
                onClick={() => navigate('/archive')}
                className="px-6 py-3 bg-red-600 text-white hover:bg-red-700 transition-colors font-semibold uppercase text-sm"
              >
                Back to Archive
              </button>
              <button
                onClick={fetchResults}
                className="px-6 py-3 border-2 border-red-600 text-red-600 hover:bg-red-50 transition-colors font-semibold uppercase text-sm"
              >
                Try Again
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!results) return null;

  const officialFormat = results.officialFormat || {};
  const sections = officialFormat.sections || {};
  const categoryScores = results.categoryScores || {};
  const hasSideViewModel = results.sideViewModelMeta !== undefined;

  return (
    <div className="min-h-screen bg-gray-50 pt-20 pb-12">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white py-12 border-b-4 border-orange-500">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-gray-400 mb-2">Classification Complete</p>
              <h1 className="text-4xl font-bold uppercase tracking-tight">View Classification Results</h1>
            </div>
            <FiCheckCircle className="w-16 h-16 text-green-400" />
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-12 space-y-8">
        
        {/* Overall Score Card */}
        {results.overallScore && (
          <div className={`border-l-4 p-8 ${getGradeBgColor(results.grade)}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-bold uppercase text-gray-600 mb-2">Overall Type Score</p>
                <div className="flex items-baseline gap-4">
                  <p className="text-5xl font-bold text-gray-900">{results.overallScore}</p>
                  <p className="text-2xl font-medium text-gray-500">/ 9.0</p>
                </div>
                <p className={`text-2xl font-bold uppercase mt-3 ${getGradeColor(results.grade)}`}>
                  {results.grade}
                </p>
              </div>
              <div className="text-right">
                <FiAward className="w-20 h-20 text-orange-500 mb-2" />
                <p className="text-sm font-bold uppercase text-gray-600">Total Traits</p>
                <p className="text-3xl font-bold text-gray-900">{results.totalTraits || 20}</p>
              </div>
            </div>
            {hasSideViewModel && (
              <div className="mt-6 pt-6 border-t-2 border-gray-300">
                <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
                  <FiActivity className="w-5 h-5 text-orange-500" />
                  <span>Side View analyzed with AI model (Real measurements included)</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Category Scores */}
        {Object.keys(categoryScores).length > 0 && (
          <div className="bg-white border-2 border-gray-200 overflow-hidden">
            <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
              <h2 className="text-2xl font-bold uppercase tracking-wide">Section Scores</h2>
              <p className="text-sm text-gray-300 mt-2">Average scores across 5 official evaluation sections</p>
            </div>
            <div className="p-8">
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(categoryScores).map(([category, score]) => (
                  <div key={category} className="bg-gray-50 border-2 border-gray-300 p-6 hover:border-orange-500 transition-colors">
                    <p className="text-xs font-bold uppercase text-gray-500 mb-3">{category}</p>
                    <p className="text-4xl font-bold text-orange-600">{score}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Detailed Trait Breakdown */}
        {Object.keys(sections).length > 0 && (
          <div className="bg-white border-2 border-gray-200 overflow-hidden">
            <div className="bg-slate-900 text-white px-8 py-6 border-b-4 border-orange-500">
              <h2 className="text-2xl font-bold uppercase tracking-wide">Detailed Trait Analysis</h2>
              <p className="text-sm text-gray-300 mt-2">Individual trait scores and measurements across all sections</p>
            </div>
            <div className="p-8 space-y-8">
              {Object.entries(sections).map(([sectionName, traits]) => (
                <div key={sectionName}>
                  <div className="flex items-center justify-between mb-4 pb-2 border-b-2 border-gray-200">
                    <h3 className="text-xl font-bold uppercase text-gray-900">{sectionName}</h3>
                    <span className="text-2xl font-bold text-orange-600">
                      {categoryScores[sectionName] || 'N/A'}
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-gray-100 border-b-2 border-gray-300">
                          <th className="text-left px-4 py-3 text-xs font-bold uppercase text-gray-600">Trait</th>
                          <th className="text-center px-4 py-3 text-xs font-bold uppercase text-gray-600">Score</th>
                          <th className="text-center px-4 py-3 text-xs font-bold uppercase text-gray-600">Measurement</th>
                        </tr>
                      </thead>
                      <tbody>
                        {traits
                          .filter(trait => trait.measurement !== null && trait.measurement !== undefined)
                          .map((trait, idx) => (
                          <tr key={idx} className="border-b border-gray-200 hover:bg-gray-50">
                            <td className="px-4 py-4 font-medium text-gray-900">{trait.trait}</td>
                            <td className="px-4 py-4 text-center">
                              <span className="inline-block px-4 py-2 bg-orange-100 text-orange-700 font-bold rounded">
                                {trait.score || 'N/A'}
                              </span>
                            </td>
                            <td className="px-4 py-4 text-center text-gray-700 font-medium">
                              {trait.measurement} cm
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Classification Info */}
        <div className="bg-white border-2 border-gray-200 p-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <p className="text-xs font-bold uppercase text-gray-500 mb-1">Classification ID</p>
              <p className="text-lg font-mono text-gray-900">{id}</p>
            </div>
            <div>
              <p className="text-xs font-bold uppercase text-gray-500 mb-1">Status</p>
              <p className="text-lg font-bold text-green-600 uppercase">{results.status || 'Completed'}</p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-4 justify-center pt-4">
          <button
            onClick={() => navigate('/classify')}
            className="px-8 py-4 bg-orange-500 text-white font-bold uppercase hover:bg-orange-600 transition-colors border-b-4 border-orange-700"
          >
            New Classification
          </button>
          <button
            onClick={() => navigate('/archive')}
            className="px-8 py-4 bg-gray-500 text-white font-bold uppercase hover:bg-gray-600 transition-colors border-b-4 border-gray-700"
          >
            View Archive
          </button>
        </div>
      </div>
    </div>
  );
};

export default ClassificationResultsPage;
