import createAxiosInstance from "./createAxiosInstance";

/**
 * Take the firmware from a GitHub release instead of the local firmware server.
 * @param {string} release "latest" (newest release with images, pre-releases included) or a tag
 * @returns {Promise<Object>} { status, message, server_url, release, release_tag, firmware_files, count }
 */
const apiUC2ConfigControllerSetOTAFirmwareRelease = async (release = "latest") => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/setOTAFirmwareRelease", {
    params: { release },
  });
  return response.data;
};

export default apiUC2ConfigControllerSetOTAFirmwareRelease;
