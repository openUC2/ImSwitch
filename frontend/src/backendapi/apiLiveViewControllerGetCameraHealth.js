import createAxiosInstance from "./createAxiosInstance";

/**
 * Per camera: { placeholder, stalled, secondsSinceNewFrame, message }.
 * placeholder = the real camera did not open and a simulated one stands in;
 * stalled = the live stream gets no new frames (camera unplugged?).
 */
const apiLiveViewControllerGetCameraHealth = async () => {
  const response = await createAxiosInstance().get("/LiveViewController/getCameraHealth");
  return response.data;
};

export default apiLiveViewControllerGetCameraHealth;
