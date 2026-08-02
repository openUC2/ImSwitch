import createAxiosInstance from "./createAxiosInstance";

// Function to move to a specific objective based on the slot
const apiObjectiveControllerMoveToObjective = async (slot, skipZ = false) => {
  try {
    const axiosInstance = createAxiosInstance(); // Create Axios instance
    const response = await axiosInstance.get(
      `/ObjectiveController/moveToObjective?slot=${slot}&skipZ=${skipZ}`,
    ); // Send GET request with the slot parameter
    const ack = response.data;

    // Explicit ACK contract: avoid treating empty/ambiguous responses as success.
    if (!ack || typeof ack !== "object" || typeof ack.accepted !== "boolean") {
      throw new Error("Objective move did not return a valid ACK object");
    }

    if (!ack.accepted) {
      const reason = ack.reason ? ` (${ack.reason})` : "";
      throw new Error(`Objective move rejected${reason}`);
    }

    return ack;
  } catch (error) {
    console.error(`Error moving to objective with slot ${slot}:`, error);
    throw error; // Throw error to be handled by the caller
  }
};

export default apiObjectiveControllerMoveToObjective;

/*
Example usage:

const moveToObjective = (slot) => {
  apiObjectiveControllerMoveToObjective(slot)
    .then((data) => {
      setMoveResult(data); // Handle success response
    })
    .catch((err) => {
      setError("Failed to move to the objective"); // Handle the error
    });
};
*/
