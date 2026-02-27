import createAxiosInstance from "./createAxiosInstance";

/**
 * Refresh overlay image for a slide using existing registration.
 * POST /ExperimentController/refreshOverviewSlideImage
 */
const apiRefreshOverviewSlideImage = async (
  slotId = "1",
  cameraName = "",
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  let url = "/ExperimentController/refreshOverviewSlideImage?";
  const params = [];
  params.push(`slot_id=${encodeURIComponent(slotId)}`);
  if (cameraName) params.push(`camera_name=${encodeURIComponent(cameraName)}`);
  if (layoutName) params.push(`layout_name=${encodeURIComponent(layoutName)}`);
  url += params.join("&");
  const response = await axiosInstance.post(url);
  return response.data;
};

export default apiRefreshOverviewSlideImage;
