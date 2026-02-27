import createAxiosInstance from "./createAxiosInstance";

/**
 * Snap overview image for a given slot.
 * POST /ExperimentController/snapOverviewImage
 */
const apiSnapOverviewImage = async (slotId = "1", cameraName = "") => {
  const axiosInstance = createAxiosInstance();
  let url = "/ExperimentController/snapOverviewImage?";
  const params = [];
  params.push(`slot_id=${encodeURIComponent(slotId)}`);
  if (cameraName) params.push(`camera_name=${encodeURIComponent(cameraName)}`);
  url += params.join("&");
  const response = await axiosInstance.post(url);
  return response.data;
};

export default apiSnapOverviewImage;
