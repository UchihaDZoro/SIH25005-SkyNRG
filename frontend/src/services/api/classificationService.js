import apiClient from './client';
import { ENDPOINTS } from './endpoints';

/**
 * Classification Service
 * All API calls for the Type Classification workflow
 */

const classificationService = {
    /**
     * Step 1: Create a new classification
     * @param {Object} animalInfo - Animal information
     * @returns {Promise} Classification ID and status
     */
    createClassification: async (animalInfo) => {
        try {
            const response = await apiClient.post(ENDPOINTS.CLASSIFICATION_CREATE, {
                animalInfo
            });
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to create classification');
        }
    },

    /**
     * Step 2: Upload 6 images for classification
     * @param {string} classificationId - Classification ID
     * @param {Array<File>} images - Array of 6 image files
     * @returns {Promise} Upload confirmation
     */
    uploadImages: async (classificationId, images) => {
        try {
            const formData = new FormData();

            // Backend expects exactly 6 images in order:
            // left_side, right_side, rear, front, udder_closeup, top_view
            images.forEach((image) => {
                formData.append('images', image);
            });

            const response = await apiClient.post(
                ENDPOINTS.CLASSIFICATION_UPLOAD_IMAGES(classificationId),
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                }
            );
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to upload images');
        }
    },

    /**
     * Step 3: Process classification with AI
     * @param {string} classificationId - Classification ID
     * @returns {Promise} Processing confirmation
     */
    processClassification: async (classificationId) => {
        try {
            const response = await apiClient.post(
                ENDPOINTS.CLASSIFICATION_PROCESS(classificationId)
            );
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to process classification');
        }
    },

    /**
     * Step 4: Get classification results
     * @param {string} classificationId - Classification ID
     * @returns {Promise} Official format results with 20 traits
     */
    getResults: async (classificationId) => {
        try {
            const response = await apiClient.get(
                ENDPOINTS.CLASSIFICATION_RESULTS(classificationId)
            );
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to get results');
        }
    },

    /**
     * Check classification status
     * @param {string} classificationId - Classification ID
     * @returns {Promise} Current status
     */
    getStatus: async (classificationId) => {
        try {
            const response = await apiClient.get(
                ENDPOINTS.CLASSIFICATION_STATUS(classificationId)
            );
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to get status');
        }
    },

    /**
     * List all classifications
     * @param {number} limit - Number of results
     * @param {number} skip - Number to skip
     * @returns {Promise} List of classifications
     */
    listClassifications: async (limit = 10, skip = 0) => {
        try {
            const response = await apiClient.get(
                `${ENDPOINTS.CLASSIFICATION_LIST}?limit=${limit}&skip=${skip}`
            );
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to list classifications');
        }
    },

    /**
     * Poll for classification status until complete
     * @param {string} classificationId - Classification ID
     * @param {function} onStatusUpdate - Callback for status updates
     * @param {number} maxAttempts - Maximum polling attempts
     * @returns {Promise} Final results
     */
    pollUntilComplete: async (classificationId, onStatusUpdate, maxAttempts = 60) => {
        let attempts = 0;

        const poll = async () => {
            const statusResponse = await classificationService.getStatus(classificationId);
            const status = statusResponse.data.status;

            if (onStatusUpdate) {
                onStatusUpdate(status);
            }

            if (status === 'completed') {
                return await classificationService.getResults(classificationId);
            } else if (status === 'failed') {
                throw new Error('Classification processing failed');
            } else if (attempts >= maxAttempts) {
                throw new Error('Classification timeout');
            }

            attempts++;
            // Wait 2 seconds before next poll
            await new Promise(resolve => setTimeout(resolve, 2000));
            return poll();
        };

        return poll();
    },

    /**
     * Get archive of all classifications with filters
     * @param {Object} params - Filter parameters (skip, limit, animal_type, breed, village, grade, search)
     * @returns {Promise} Paginated archive results
     */
    getArchive: async (params = {}) => {
        try {
            const queryParams = new URLSearchParams();

            // Add all provided parameters
            Object.keys(params).forEach(key => {
                if (params[key] !== null && params[key] !== undefined && params[key] !== '') {
                    queryParams.append(key, params[key]);
                }
            });

            const url = `${ENDPOINTS.CLASSIFICATION_ARCHIVE}${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
            const response = await apiClient.get(url);
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to fetch archive');
        }
    },

    /**
     * Delete a classification by ID
     * @param {string} classificationId - Classification ID to delete
     * @returns {Promise} Delete confirmation
     */
    deleteClassification: async (classificationId) => {
        try {
            const response = await apiClient.delete(
                ENDPOINTS.CLASSIFICATION_DELETE(classificationId)
            );
            return response;
        } catch (error) {
            throw new Error(error.response?.data?.detail || 'Failed to delete classification');
        }
    }
};

export default classificationService;
