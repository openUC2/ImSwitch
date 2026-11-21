// src/backendapi/apiFileManagerGetDatasets.js
import createAxiosInstance from "./createAxiosInstance";

/**
 * OME-Zarr stores and OME-TIFF files below a folder of the data path, newest first.
 * Does not walk into zarr stores (the File Manager listing does).
 *
 * @param {string} path - folder relative to the data path ("" = all of it)
 * @returns {Promise<Array<{name, path, folder, format, modifiedTime}>>}
 */
const apiFileManagerGetDatasets = async (path = "") => {
  const response = await createAxiosInstance().get("/FileManager/datasets", { params: { path } });
  return response.data;
};

export default apiFileManagerGetDatasets;
