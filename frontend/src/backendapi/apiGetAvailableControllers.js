import createAxiosInstance from "./createAxiosInstance";

const apiGetAvailableControllers = async () => {
  try {
    const axiosInstance = createAxiosInstance();
    const response = await axiosInstance.get("/getAvailableControllers");
    const controllers = response?.data?.availableControllers;
    return Array.isArray(controllers) ? controllers : [];
  } catch (error) {
    console.warn(
      "Failed to fetch available controllers:",
      error?.message || error,
    );
    throw error;
  }
};
};

export default apiGetAvailableControllers;
