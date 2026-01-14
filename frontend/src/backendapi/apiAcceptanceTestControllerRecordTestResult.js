/**
 * API call to record a test result
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Record a test result
 * @param {string} category - Test category (motion, lighting, camera, autofocus)
 * @param {string} testName - Name of the specific test
 * @param {boolean} passed - Whether the test passed
 * @param {string} notes - Optional notes from user
 * @returns {Promise} Response with confirmation
 */
export default async function apiAcceptanceTestControllerRecordTestResult(
  category,
  testName,
  passed,
  notes = ''
) {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/recordTestResult', {
    params: { category, test_name: testName, passed, notes }
  });
  return response.data;
}
