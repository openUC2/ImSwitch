// src/backendapi/apiLiveViewControllerGetLongExposureInfo.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * Ask the backend whether the detector's exposure rules out live streaming.
 * GET /LiveViewController/getLongExposureInfo
 *
 * At multi-second exposures the camera delivers a frame slower than the stream
 * worker's grab timeout, so the preview only shows stale frames. The backend
 * refuses to start a stream above `thresholdMs` (unless forced); the UI uses
 * this to skip auto-start on load and to switch to snap-on-demand instead.
 *
 * @param {string|null} detectorName - Detector to query (null = current detector)
 *
 * Returns: { detector, exposureMs, thresholdMs, isLongExposure }
 */
const apiLiveViewControllerGetLongExposureInfo = async (detectorName = null) => {
  const fallback = {
    detector: detectorName,
    exposureMs: 0,
    thresholdMs: 2000,
    isLongExposure: false,
  };
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/getLongExposureInfo`;
    const params = {};
    if (detectorName !== null && detectorName !== undefined) {
      params.detectorName = detectorName;
    }
    const response = await axiosInstance.get(url, { params });
    if (response.data && typeof response.data === "object") {
      return { ...fallback, ...response.data };
    }
    return fallback;
  } catch (error) {
    // Older backends don't have this endpoint — treat as "not long exposure"
    // so the stream behaves exactly as it did before.
    console.warn("Failed to get long-exposure info:", error.message);
    return fallback;
  }
};

export default apiLiveViewControllerGetLongExposureInfo;

/**
 * Example usage:
 * const info = await apiLiveViewControllerGetLongExposureInfo();
 * if (info.isLongExposure) { /* offer Snap instead of Start *\/ }
 */
