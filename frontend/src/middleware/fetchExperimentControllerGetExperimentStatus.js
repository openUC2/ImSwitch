import * as expermentStatusSlice from "../state/slices/ExperimentStatusSlice.js";
 
import apiGetExperimentStatus from "../backendapi/apiExperimentControllerGetExperimentStatus.js";


const fetchExperimentControllerGetExperimentStatus = (dispatch) => {
    //request
    apiGetExperimentStatus()
    .then((data) => {
      console.log("apiGetExperimentStatus", data);

      // Update Redux
      dispatch(expermentStatusSlice.setStatus(data.status));
    })
    .catch((err) => {
      console.error("Failed to fetch experiment parameters", err);
    });
};

export default fetchExperimentControllerGetExperimentStatus;
