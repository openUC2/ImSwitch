// Home a specific positioner axis
import createAxiosInstance from './createAxiosInstance';

const apiPositionerControllerHomeAxis = async ({
  positionerName = null,
  axis = "X",
  isBlocking = false,
  homeDirection = null,
  homeSpeed = null,
  homeEndstoppolarity = null,
  homeEndposRelease = null,
  homeTimeout = null
}) => {
  const axiosInstance = createAxiosInstance();
  const params = {
    axis,
    isBlocking
  };
  
  if (positionerName) params.positionerName = positionerName;
  if (homeDirection !== null) params.homeDirection = homeDirection;
  if (homeSpeed !== null) params.homeSpeed = homeSpeed;
  if (homeEndstoppolarity !== null) params.homeEndstoppolarity = homeEndstoppolarity;
  if (homeEndposRelease !== null) params.homeEndposRelease = homeEndposRelease;
  if (homeTimeout !== null) params.homeTimeout = homeTimeout;
  
  const response = await axiosInstance.get('/PositionerController/homeAxis', {
    params
  });
  return response.data;
};

export default apiPositionerControllerHomeAxis;
