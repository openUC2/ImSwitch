/**
 * API wrapper for AcceptanceTestController.setLaserActiveAcceptanceTest
 * Sets a laser/light source active or inactive during acceptance testing
 */

import createAxiosInstance from './createAxiosInstance';

export default async function apiAcceptanceTestControllerSetLaserActive(laser, active, intensity = 255) {
  const axiosInstance = createAxiosInstance();
  const params = {
    laser: laser,
    active: active,
    intensity: intensity
  };
  
  const response = await axiosInstance.get('/AcceptanceTestController/setLaserActiveAcceptanceTest', { params });
  return response.data;
}
