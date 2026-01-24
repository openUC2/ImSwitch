/**
 * API call to run autofocus
 */
import createAxiosInstance from './createAxiosInstance';

/**
 * Run software-based autofocus
 * @returns {Promise} Response with autofocus result
 */
export default async function apiAcceptanceTestControllerRunAutofocus() {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/AcceptanceTestController/runAutofocus');
  return response.data;
}
