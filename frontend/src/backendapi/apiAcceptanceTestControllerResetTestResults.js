/**
 * API call to reset test results
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Reset all test results to start a new test session
 * @returns {Promise} Response with confirmation
 */
export default async function apiAcceptanceTestControllerResetTestResults() {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/resetTestResults');
  return response.data;
}
