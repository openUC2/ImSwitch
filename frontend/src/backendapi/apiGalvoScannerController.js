/**
 * Galvo Scanner API functions
 * 
 * Provides API functions for communicating with the GalvoScannerController backend.
 * Follows the same pattern as other API modules in the frontend.
 */

/**
 * Build API base URL
 */
const getApiBase = (hostIP, hostPort) => 
  `${hostIP}:${hostPort}/imswitch/api/GalvoScannerController`;

/**
 * Get all galvo scanner device names
 */
export const apiGetGalvoScannerNames = async (hostIP, hostPort) => {
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getGalvoScannerNames`);
  return response.json();
};

/**
 * Get configuration for a scanner
 */
export const apiGetGalvoScannerConfig = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getGalvoScannerConfig${params}`);
  return response.json();
};

/**
 * Get status of a scanner
 */
export const apiGetGalvoScannerStatus = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getGalvoScannerStatus${params}`);
  return response.json();
};

/**
 * Set scan configuration (without starting)
 */
export const apiSetGalvoScanConfig = async (hostIP, hostPort, scannerName, config) => {
  const params = new URLSearchParams({ scannerName });
  
  // Add config parameters
  Object.entries(config).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      params.append(key, String(value));
    }
  });
  
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/setGalvoScanConfig?${params}`);
  return response.json();
};

/**
 * Start galvo scan with given configuration
 */
export const apiStartGalvoScan = async (hostIP, hostPort, scannerName, config) => {
  const params = new URLSearchParams({ scannerName });
  
  // Add config parameters
  Object.entries(config).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      params.append(key, String(value));
    }
  });
  
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/startGalvoScan?${params}`);
  return response.json();
};

/**
 * Stop galvo scan
 */
export const apiStopGalvoScan = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/stopGalvoScan${params}`);
  return response.json();
};

/**
 * Get status of all scanners
 */
export const apiGetAllGalvoScannersStatus = async (hostIP, hostPort) => {
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getAllGalvoScannersStatus`);
  return response.json();
};

/**
 * Stop all galvo scans
 */
export const apiStopAllGalvoScans = async (hostIP, hostPort) => {
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/stopAllGalvoScans`);
  return response.json();
};

// ========================
// Arbitrary Points API
// ========================

/**
 * Set arbitrary points for scanning
 * @param {Array} points - Array of {x, y, dwell_us, laser_intensity}
 * @param {boolean} applyAffine - Whether to apply camera->galvo affine transform
 */
export const apiSetArbitraryPoints = async (hostIP, hostPort, scannerName, points, laserTrigger = 'AUTO', applyAffine = true) => {
  const params = new URLSearchParams({
    scannerName,
    points: JSON.stringify(points),
    laser_trigger: laserTrigger,
    apply_affine: String(applyAffine),
  });
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/setArbitraryPoints?${params}`);
  return response.json();
};

/**
 * Start arbitrary point scanning
 */
export const apiStartArbitraryScan = async (hostIP, hostPort, scannerName, points, laserTrigger = 'AUTO', applyAffine = true) => {
  const params = new URLSearchParams({
    scannerName,
    points: JSON.stringify(points),
    laser_trigger: laserTrigger,
    apply_affine: String(applyAffine),
  });
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/startArbitraryScan?${params}`);
  return response.json();
};

/**
 * Stop arbitrary point scanning
 */
export const apiStopArbitraryScan = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/stopArbitraryScan${params}`);
  return response.json();
};

/**
 * Pause arbitrary point scanning
 */
export const apiPauseArbitraryScan = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/pauseArbitraryScan${params}`);
  return response.json();
};

/**
 * Resume arbitrary point scanning
 */
export const apiResumeArbitraryScan = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/resumeArbitraryScan${params}`);
  return response.json();
};

/**
 * Get arbitrary scan state
 */
export const apiGetArbitraryScanState = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getArbitraryScanState${params}`);
  return response.json();
};

// ========================
// Affine Transform API
// ========================

/**
 * Get current affine transform
 */
export const apiGetAffineTransform = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getAffineTransform${params}`);
  return response.json();
};

/**
 * Set affine transform parameters
 */
export const apiSetAffineTransform = async (hostIP, hostPort, scannerName, affine, save = true) => {
  const params = new URLSearchParams({ scannerName, save: String(save) });
  Object.entries(affine).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      params.append(key, String(value));
    }
  });
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/setAffineTransform?${params}`);
  return response.json();
};

/**
 * Reset affine transform to identity
 */
export const apiResetAffineTransform = async (hostIP, hostPort, scannerName = null, save = true) => {
  const params = new URLSearchParams();
  if (scannerName) params.append('scannerName', scannerName);
  params.append('save', String(save));
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/resetAffineTransform?${params}`);
  return response.json();
};

/**
 * Run affine calibration from point pairs
 */
export const apiRunAffineCalibration = async (hostIP, hostPort, scannerName, camPoints, galvoPoints, save = true) => {
  const calibrationData = JSON.stringify({
    cam_points: camPoints,
    galvo_points: galvoPoints,
  });
  const params = new URLSearchParams({
    scannerName,
    calibration_data: calibrationData,
    save: String(save),
  });
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/runAffineCalibration?${params}`);
  return response.json();
};

/**
 * Get calibration points suggestion
 */
export const apiGetCalibrationPoints = async (hostIP, hostPort, scannerName = null) => {
  const params = scannerName ? `?scannerName=${scannerName}` : '';
  const response = await fetch(`${getApiBase(hostIP, hostPort)}/getCalibrationPoints${params}`);
  return response.json();
};

export default {
  apiGetGalvoScannerNames,
  apiGetGalvoScannerConfig,
  apiGetGalvoScannerStatus,
  apiSetGalvoScanConfig,
  apiStartGalvoScan,
  apiStopGalvoScan,
  apiGetAllGalvoScannersStatus,
  apiStopAllGalvoScans,
  apiSetArbitraryPoints,
  apiStartArbitraryScan,
  apiStopArbitraryScan,
  apiPauseArbitraryScan,
  apiResumeArbitraryScan,
  apiGetArbitraryScanState,
  apiGetAffineTransform,
  apiSetAffineTransform,
  apiResetAffineTransform,
  apiRunAffineCalibration,
  apiGetCalibrationPoints,
};
