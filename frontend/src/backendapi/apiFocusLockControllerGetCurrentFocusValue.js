import createAxiosInstance from './createAxiosInstance';

/**
 * Get current focus value and metadata
 * @returns {Promise} Promise containing focus value, timestamp, and lock status
 */
export default async function apiFocusLockControllerGetCurrentFocusValue() {
  const axios = createAxiosInstance();
  try {
    const response = await axios.get('/FocusLockController/getCurrentFocusValue');
    return response.data;
  } catch (error) {
    console.error('Failed to get current focus value:', error);
    throw error;
  }
}
