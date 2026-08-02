import createAxiosInstance from "./createAxiosInstance";

// Set the motor speed used when switching between objective slots
const apiObjectiveControllerSetMoveSpeed = async (speed) => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get(
      `/ObjectiveController/setMoveSpeed?speed=${speed}`
    );
    return response.data;
  } catch (error) {
    console.error(`Error setting objective move speed to ${speed}:`, error);
    throw error;
  }
};

export default apiObjectiveControllerSetMoveSpeed;
