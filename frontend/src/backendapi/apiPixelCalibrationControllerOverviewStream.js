// Start/stop the overview camera stream
// Note: overviewStream returns an MJPEG StreamingResponse, not JSON
// Use the stream URL directly in <img> src for the stream
// This function is for stopping the stream only
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerOverviewStream = async (startStream = true) => {
  const axiosInstance = createAxiosInstance();
  
  // Only call the endpoint to stop the stream
  // For starting, just use the stream URL directly in img src
  if (!startStream) {
    const response = await axiosInstance.get('/PixelCalibrationController/overviewStream', {
      params: { startStream: false }
    });
    return response.data;
  }
  
  // For starting, return success without calling (stream starts when accessed)
  return { status: 'success', message: 'Stream will start when accessed' };
};

export default apiPixelCalibrationControllerOverviewStream;
