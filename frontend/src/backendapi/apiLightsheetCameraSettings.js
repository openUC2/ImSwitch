// API functions for lightsheet camera settings (exposure, gain)
import store from "../state/store.js";

/**
 * Get current camera exposure time
 */
export async function apiGetCameraExposureTime() {
  const state = store.getState();
  const { ip, apiPort } = state.connectionSettingsState;
  
  const url = `${ip}:${apiPort}/imswitch/api/LightsheetController/getCameraExposureTime`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error getting camera exposure time:", error);
    throw error;
  }
}

/**
 * Set camera exposure time
 * @param {number} exposureTime - Exposure time in milliseconds
 */
export async function apiSetCameraExposureTime(exposureTime) {
  const state = store.getState();
  const { ip, apiPort } = state.connectionSettingsState;
  
  const url = `${ip}:${apiPort}/imswitch/api/LightsheetController/setCameraExposureTime?exposureTime=${exposureTime}`;
  
  try {
    const response = await fetch(url, { method: "GET" });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error setting camera exposure time:", error);
    throw error;
  }
}

/**
 * Get current camera gain
 */
export async function apiGetCameraGain() {
  const state = store.getState();
  const { ip, apiPort } = state.connectionSettingsState;
  
  const url = `${ip}:${apiPort}/imswitch/api/LightsheetController/getCameraGain`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error getting camera gain:", error);
    throw error;
  }
}

/**
 * Set camera gain
 * @param {number} gain - Camera gain value
 */
export async function apiSetCameraGain(gain) {
  const state = store.getState();
  const { ip, apiPort } = state.connectionSettingsState;
  
  const url = `${ip}:${apiPort}/imswitch/api/LightsheetController/setCameraGain?gain=${gain}`;
  
  try {
    const response = await fetch(url, { method: "GET" });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error setting camera gain:", error);
    throw error;
  }
}
