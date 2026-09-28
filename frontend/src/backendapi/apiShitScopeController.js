// ShitScopeController: dedicated tile scan with live preview, registration and stitching.
import createAxiosInstance from "./createAxiosInstance";

const get = async (method, params = {}, timeout) => {
  const response = await createAxiosInstance().get(`/ShitScopeController/${method}`, { params, timeout });
  return response.data;
};

// Pixel size, FOV, stage hints, suggested steps (whole full steps, min overlap), status
export const apiShitScopeGetInfo = () => get("getShitScopeInfo");
// homeFirst: drive into the -X/-Y stop before the first tile. Steps of 0 = suggested; pattern "" = stage recommendation (raster for the PCB stage)
export const apiShitScopeStartScan = ({ nx, ny, stepXUm = 0, stepYUm = 0, pattern = "", centered = true, returnToStart = true, homeFirst = false,
  overlap = 0, speed = 0, settleMs = 0 }) =>
  get("startShitScopeScan", { nx, ny, stepXUm, stepYUm, pattern, centered, returnToStart, homeFirst, overlap, speed, settleMs });
export const apiShitScopeStopScan = () => get("stopShitScopeScan");
export const apiShitScopeGetStatus = () => get("getShitScopeStatus");
export const apiShitScopeGetPreview = () => get("getShitScopePreview");
export const apiShitScopeGetResult = () => get("getShitScopeResult");
export const apiShitScopeAnalyzeScan = (scanDir = "") => get("analyzeShitScopeScan", { scanDir }, 300000);
// Full-resolution (colour if the camera is colour) stitch of the last scan -> stitched_full.tif
export const apiShitScopeExportFullRes = (scanDir = "") => get("exportShitScopeFullRes", { scanDir }, 600000);
