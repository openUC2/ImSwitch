/**
 * API call to get test report
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Get complete test report with all results
 * @returns {Promise} Response with all test results
 */
export default async function apiAcceptanceTestControllerGetTestReport() {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/getTestReport');
  return response.data;
}
