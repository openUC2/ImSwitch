// Read-noise calibration: write/replace the free-text comment for a session
import createAxiosInstance from './createAxiosInstance';

const apiReadNoiseCalibrationControllerSaveComment = async (sessionId, comment) => {
  const axiosInstance = createAxiosInstance();
  // sessionId goes as a query param; comment is sent in the JSON body (embed=True)
  const response = await axiosInstance.post(
    '/ReadNoiseCalibrationController/saveComment',
    { comment },
    { params: { sessionId } }
  );
  return response.data;
};

export default apiReadNoiseCalibrationControllerSaveComment;
