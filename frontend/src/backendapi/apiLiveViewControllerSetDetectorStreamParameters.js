import createAxiosInstance from "./createAxiosInstance";

const apiLiveViewControllerSetDetectorStreamParameters = async (detectorName, params) => {
  try {
    const axiosInstance = createAxiosInstance();
    const url = `/LiveViewController/setDetectorStreamParameters`;

    const response = await axiosInstance.post(url, params, {
      params: { detectorName },
      headers: {
        'Content-Type': 'application/json',
      }
    });

    return response.data;
  } catch (error) {
    console.error('Error setting detector stream parameters:', error);
    throw error;
  }
};

export default apiLiveViewControllerSetDetectorStreamParameters;
