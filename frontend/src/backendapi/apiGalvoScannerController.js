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

export default {
  apiGetGalvoScannerNames,
  apiGetGalvoScannerConfig,
  apiGetGalvoScannerStatus,
  apiSetGalvoScanConfig,
  apiStartGalvoScan,
  apiStopGalvoScan,
  apiGetAllGalvoScannersStatus,
  apiStopAllGalvoScans,
};
