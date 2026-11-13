import createAxiosInstance from "./createAxiosInstance";

/**
 * GitHub releases of the firmware repository, newest first.
 * @returns {Promise<Object>} { status, repo, releases: [{ tag, name, prerelease, published_at, url, image_count }] }
 *   image_count 0 = only web-flasher images, nothing ImSwitch can flash
 */
const apiUC2ConfigControllerListFirmwareReleases = async () => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get("/UC2ConfigController/listFirmwareReleases");
  return response.data;
};

export default apiUC2ConfigControllerListFirmwareReleases;
