import createAxiosInstance from "./createAxiosInstance";

/**
 * Load a single slot's stored JPEG overlay (base64-encoded).
 * GET /ExperimentController/getOverviewOverlayImage
 */
const apiGetOverviewOverlayImage = async (
  slotId,
  cameraName = "",
  layoutName = "Heidstar 4x Histosample"
) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get(
    "/ExperimentController/getOverviewOverlayImage",
    {
      params: {
        slot_id: slotId,
        camera_name: cameraName,
        layout_name: layoutName,
      },
    }
  );
  return response.data;
};

export default apiGetOverviewOverlayImage;
