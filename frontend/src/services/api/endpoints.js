// API Base URL
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const API_VERSION = '/api/v1';

// Classification Endpoints
export const ENDPOINTS = {
    // Classification
    CLASSIFICATION_CREATE: `${API_VERSION}/classification/create`,
    CLASSIFICATION_UPLOAD_IMAGES: (id) => `${API_VERSION}/classification/${id}/upload-images`,
    CLASSIFICATION_PROCESS: (id) => `${API_VERSION}/classification/${id}/process`,
    CLASSIFICATION_RESULTS: (id) => `${API_VERSION}/classification/${id}/results`,
    CLASSIFICATION_STATUS: (id) => `${API_VERSION}/classification/${id}/status`,
    CLASSIFICATION_LIST: `${API_VERSION}/classification/list`,
    CLASSIFICATION_ARCHIVE: `${API_VERSION}/classification/archive`,
    CLASSIFICATION_DELETE: (id) => `${API_VERSION}/classification/${id}`,

    // Health
    ROOT: '/',
    HEALTH: '/health',
};
