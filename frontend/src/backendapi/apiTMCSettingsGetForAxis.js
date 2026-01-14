/**
 * API call to get TMC stepper driver settings for a specific motor axis from the device
 * 
 * @param {string} baseURL - The base URL for the API
 * @param {string} axis - The motor axis (X, Y, Z, or A)
 * @returns {Promise<Object>} Response with TMC settings
 */
async function apiTMCSettingsGetForAxis(baseURL, axis) {
  const axios = (await import('./createAxiosInstance')).default(baseURL);
  
  try {
    const response = await axios.get('/uc2config/getTMCSettingsForAxis', {
      params: { axis }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error getting TMC settings for axis:', error);
    throw error;
  }
}

export default apiTMCSettingsGetForAxis;
