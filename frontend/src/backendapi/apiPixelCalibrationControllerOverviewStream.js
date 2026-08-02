// Overview camera stream control via LiveViewController.
// The ObservationCamera MJPEG stream is exposed at:
//   GET /LiveViewController/mjpeg_stream?detectorName=ObservationCamera
// Use that URL directly in an <img> src for display.
// This helper starts or stops the stream via LiveViewController.
import createAxiosInstance from './createAxiosInstance';

const apiPixelCalibrationControllerOverviewStream = async (startStream = true) => {
  const axiosInstance = createAxiosInstance();

  if (startStream) {
    // Start MJPEG stream for ObservationCamera with 1x subsampling
    const response = await axiosInstance.post(
      '/LiveViewController/startLiveView',
      { subsampling_factor: 1 },
      { params: { detectorName: 'ObservationCamera', protocol: 'mjpeg' } }
    );
    return response.data;
  } else {
    // Stop the ObservationCamera stream
    const response = await axiosInstance.get('/LiveViewController/stopLiveView', {
      params: { detectorName: 'ObservationCamera' },
    });
    return response.data;
  }
};

export default apiPixelCalibrationControllerOverviewStream;
