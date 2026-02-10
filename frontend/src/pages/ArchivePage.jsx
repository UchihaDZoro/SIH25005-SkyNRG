import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FiSearch, FiFilter, FiDownload, FiClock, FiAward, FiChevronLeft, FiChevronRight, FiTrash2, FiEye } from 'react-icons/fi';
import { classificationService } from '../services/api';
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

const ArchivePage = () => {
  const [classifications, setClassifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [totalResults, setTotalResults] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  const [filters, setFilters] = useState({
    search: '',
    animal_type: '',
    grade: '',
    skip: 0,
    limit: 20
  });

  useEffect(() => {
    fetchArchive();
  }, [filters]);

  const fetchArchive = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await classificationService.getArchive(filters);
      setClassifications(response.data.results);
      setTotalResults(response.data.total);
      setCurrentPage(response.data.page);
      setTotalPages(response.data.totalPages);
    } catch (err) {
      setError(err.message || 'Failed to load archive');
      console.error('Archive fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchChange = (e) => {
    setFilters(prev => ({
      ...prev,
      search: e.target.value,
      skip: 0
    }));
  };

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({
      ...prev,
      [key]: value,
      skip: 0
    }));
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setFilters(prev => ({
        ...prev,
        skip: (newPage - 1) * prev.limit
      }));
    }
  };

  const handleDelete = async (id) => {
    try {
      await classificationService.deleteClassification(id);
      setDeleteConfirm(null);
      fetchArchive();
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Failed to delete classification: ' + err.message);
    }
  };

  const exportToExcel = () => {
    if (classifications.length === 0) return;

    try {
      const worksheetData = [
        ['Classification ID', 'Overall Score', 'Grade', 'Confidence', 'Classification Date'],
        ...classifications.map(c => [
          c.id,
          c.overallScore,
          c.grade,
          c.confidenceLevel,
          new Date(c.createdAt).toLocaleDateString()
        ])
      ];

      const worksheet = XLSX.utils.aoa_to_sheet(worksheetData);
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, 'Classifications');

      const filename = `classification_archive_${new Date().toISOString().split('T')[0]}.xlsx`;
      const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
      const blob = new Blob([excelBuffer], { 
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
      });
      
      saveAs(blob, filename);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export Excel file. Please try again.');
    }
  };

  const getGradeColor = (grade) => {
    const colors = {
      'Excellent': 'text-green-600 bg-green-50',
      'Good': 'text-blue-600 bg-blue-50',
      'Fair': 'text-yellow-700 bg-yellow-50',
      'Poor': 'text-red-600 bg-red-50'
    };
    return colors[grade] || 'text-gray-600 bg-gray-50';
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-20">
      {/* Header */}
      <section className="bg-gradient-to-r from-slate-900 to-slate-800 text-white py-12 border-b-4 border-orange-500">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-bold tracking-widest uppercase text-gray-400 mb-2">Classification Management</p>
              <h1 className="text-4xl font-bold tracking-tight">Archive</h1>
            </div>
            <button
              onClick={exportToExcel}
              disabled={classifications.length === 0}
              className="px-6 py-3 bg-orange-500 text-white font-semibold uppercase text-sm tracking-wide hover:bg-orange-600 transition-colors inline-flex items-center gap-2 disabled:bg-gray-600 disabled:cursor-not-allowed border-b-2 border-orange-700"
            >
              <FiDownload className="w-4 h-4" />
              Export Excel
            </button>
          </div>
        </div>
      </section>

      {/* Search and Filters */}
      <section className="bg-white border-b-2 border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-grow">
              <FiSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search by classification ID..."
                value={filters.search}
                onChange={handleSearchChange}
                className="w-full pl-12 pr-4 py-3 border-2 border-gray-300 focus:border-orange-500 outline-none transition-all text-sm font-medium"
              />
            </div>
            <select
              value={filters.grade}
              onChange={(e) => handleFilterChange('grade', e.target.value)}
              className="px-4 py-3 border-2 border-gray-300 bg-white focus:border-orange-500 outline-none transition-all text-sm font-semibold min-w-[180px]"
            >
              <option value="">ALL GRADES</option>
              <option value="Excellent">EXCELLENT</option>
              <option value="Good">GOOD</option>
              <option value="Fair">FAIR</option>
              <option value="Poor">POOR</option>
            </select>
            {(filters.search || filters.grade) && (
              <button
                onClick={() => setFilters({ search: '', animal_type: '', grade: '', skip: 0, limit: 20 })}
                className="px-6 py-3 border-2 border-gray-300 text-gray-700 font-semibold uppercase text-sm hover:bg-gray-50 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
          <div className="mt-4 text-xs font-bold text-gray-500 uppercase tracking-wider">
            {totalResults} Total Classifications
          </div>
        </div>
      </section>

      {/* Table Section */}
      <section className="py-8">
        <div className="max-w-7xl mx-auto px-6">
          {loading ? (
            <div className="text-center py-24">
              <div className="inline-block w-16 h-16 border-4 border-gray-300 border-t-orange-500 animate-spin"></div>
              <p className="mt-6 text-gray-600 font-semibold uppercase text-sm">Loading...</p>
            </div>
          ) : error ? (
            <div className="bg-red-50 border-l-4 border-red-500 p-8 text-center">
              <p className="text-red-700 font-semibold">{error}</p>
              <button
                onClick={fetchArchive}
                className="mt-4 px-6 py-3 bg-red-600 text-white font-semibold hover:bg-red-700 transition-colors uppercase text-sm"
              >
                Retry
              </button>
            </div>
          ) : classifications.length === 0 ? (
            <div className="text-center py-24 bg-white border-2 border-gray-200">
              <div className="text-6xl mb-6">📊</div>
              <h3 className="text-2xl font-bold text-gray-800 mb-2 uppercase tracking-wide">No Classifications Found</h3>
              <p className="text-gray-500 font-medium">Try adjusting your search or filters</p>
            </div>
          ) : (
            <>
              {/* Table */}
              <div className="bg-white border-2 border-gray-200 overflow-x-auto">
                <table className="w-full min-w-[800px]">
                  <thead className="bg-slate-900 text-white">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider border-r border-slate-700">Classification ID</th>
                      <th className="px-6 py-4 text-center text-xs font-bold uppercase tracking-wider border-r border-slate-700">Score</th>
                      <th className="px-6 py-4 text-center text-xs font-bold uppercase tracking-wider border-r border-slate-700">Grade</th>
                      <th className="px-6 py-4 text-center text-xs font-bold uppercase tracking-wider border-r border-slate-700">Date</th>
                      <th className="px-6 py-4 text-center text-xs font-bold uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {classifications.map((classification, index) => (
                      <tr 
                        key={classification.id} 
                        className={`border-b-2 border-gray-200 hover:bg-gray-50 transition-colors ${
                          index % 2 === 0 ? 'bg-white' : 'bg-gray-50'
                        }`}
                      >
                        <td className="px-6 py-4 border-r border-gray-200">
                          <div className="font-mono text-xs text-gray-700 break-all">{classification.id}</div>
                        </td>
                        <td className="px-6 py-4 text-center border-r border-gray-200">
                          <div className="text-3xl font-bold text-gray-900">{classification.overallScore}</div>
                        </td>
                        <td className="px-6 py-4 text-center border-r border-gray-200">
                          <span className={`px-4 py-2 font-bold text-xs uppercase tracking-wider inline-block ${getGradeColor(classification.grade)}`}>
                            {classification.grade}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center border-r border-gray-200">
                          <div className="flex items-center justify-center gap-2 text-sm text-gray-600 font-medium">
                            <FiClock className="w-4 h-4" />
                            {formatDate(classification.createdAt)}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex flex-col sm:flex-row items-center justify-center gap-2">
                            <Link
                              to={`/classification/${classification.id}`}
                              className="px-4 py-2 bg-orange-500 text-white font-semibold text-xs uppercase hover:bg-orange-600 transition-colors inline-flex items-center gap-2"
                            >
                              <FiEye className="w-4 h-4" />
                              View
                            </Link>
                            <button
                              onClick={() => setDeleteConfirm(classification.id)}
                              className="px-4 py-2 bg-red-500 text-white font-semibold text-xs uppercase hover:bg-red-600 transition-colors inline-flex items-center gap-2"
                            >
                              <FiTrash2 className="w-4 h-4" />
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-2 mt-8">
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="p-3 border-2 border-gray-300 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <FiChevronLeft className="w-5 h-5" />
                  </button>

                  {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                    let pageNum;
                    if (totalPages <= 7) {
                      pageNum = i + 1;
                    } else if (currentPage <= 4) {
                      pageNum = i + 1;
                    } else if (currentPage >= totalPages - 3) {
                      pageNum = totalPages - 6 + i;
                    } else {
                      pageNum = currentPage - 3 + i;
                    }

                    return (
                      <button
                        key={pageNum}
                        onClick={() => handlePageChange(pageNum)}
                        className={`px-5 py-3 border-2 font-bold text-sm transition-colors ${
                          currentPage === pageNum
                            ? 'bg-orange-500 text-white border-orange-500'
                            : 'border-gray-300 hover:bg-gray-100 text-gray-700'
                        }`}
                      >
                        {pageNum}
                      </button>
                    );
                  })}

                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                    className="p-3 border-2 border-gray-300 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <FiChevronRight className="w-5 h-5" />
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-6">
          <div className="bg-white max-w-md w-full p-8 border-4 border-gray-200">
            <h3 className="text-2xl font-bold text-gray-900 mb-4 uppercase tracking-wide">Delete Classification?</h3>
            <p className="text-gray-600 mb-6 font-medium">
              Are you sure you want to delete this classification? This action cannot be undone.
            </p>
            <div className="flex gap-4">
              <button
                onClick={() => setDeleteConfirm(null)}
                className="flex-1 px-6 py-3 border-2 border-gray-300 text-gray-700 font-bold uppercase text-sm hover:bg-gray-100 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(deleteConfirm)}
                className="flex-1 px-6 py-3 bg-red-500 text-white font-bold uppercase text-sm hover:bg-red-600 transition-colors inline-flex items-center justify-center gap-2"
              >
                <FiTrash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ArchivePage;
