// Associate laser name to channel index
import createAxiosInstance from './createAxiosInstance';

const apiLaserControllerSetLaserChannelIndex = async (laserName, channelIndex) => {
  const axiosInstance = createAxiosInstance();
  const response = await axiosInstance.get('/LaserController/setLaserChannelIndex', {
    params: {
      laserName,
      channelIndex
    }
  });
  return response.data;
};

export default apiLaserControllerSetLaserChannelIndex;
