import apiObjectiveControllerGetStatus from "../backendapi/apiObjectiveControllerGetStatus.js";

import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";

const fetchObjectiveControllerGetStatus = (dispatch) => {
  // Request to fetch objective status
  apiObjectiveControllerGetStatus()
    .then((config) => {
      console.log("fetchObjectiveControllerGetStatus", config);
      // TODO: THIS IS CALLED TOO OFTEN! 
      // Update Redux with the fetched status
      dispatch(objectiveSlice.setFovX(config.FOV[0]*1)); //* 1000)); //TODO remove fov dummy config correction factor
      dispatch(objectiveSlice.setFovY(config.FOV[1]*1));// * 1000)); //TODO remove fov dummy config correction factor

      //TODO check if field exists
      // Backend returns x0, x1, z0, z1 (not x1, x2, z1, z2)
      dispatch(objectiveSlice.setPosX0(config.x0));
      dispatch(objectiveSlice.setPosX1(config.x1)); 
      dispatch(objectiveSlice.setPixelSize(config.pixelsize)); 
      dispatch(objectiveSlice.setPosZ0(config.z0));
      dispatch(objectiveSlice.setPosZ1(config.z1));
      dispatch(objectiveSlice.setMagnification(config.magnification))
      dispatch(objectiveSlice.setNA(config.NA))
      dispatch(objectiveSlice.setmagnification1(config.availableObjectivesNames[0]))
      dispatch(objectiveSlice.setmagnification2(config.availableObjectivesNames[1]))
  
      //TODO add more objective results
      /*reposne: 
      {
        "x1": 0,
        "x2": 0,
        "pos": 0,
        "isHomed": 0,
        "state": 0,
        "isRunning": 0,
        "FOV": [
            100,
            100
        ],
        "pixelsize": 0.2
      }
      */
    })
    .catch((err) => {
      console.error("Failed to fetch objective status", err);
    });
};

export default fetchObjectiveControllerGetStatus;
