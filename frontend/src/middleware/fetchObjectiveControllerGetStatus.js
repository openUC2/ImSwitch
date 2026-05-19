import apiObjectiveControllerGetStatus from "../backendapi/apiObjectiveControllerGetStatus.js";

import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";

const fetchObjectiveControllerGetStatus = (dispatch) => {
  apiObjectiveControllerGetStatus()
    .then((config) => {
      console.log("fetchObjectiveControllerGetStatus", config);

      if (config.FOV) {
        dispatch(objectiveSlice.setFovX(config.FOV[0]));
        dispatch(objectiveSlice.setFovY(config.FOV[1]));
      }

      if (config.x0 != null) dispatch(objectiveSlice.setPosX0(config.x0));
      if (config.x1 != null) dispatch(objectiveSlice.setPosX1(config.x1));
      if (config.z0 != null) dispatch(objectiveSlice.setPosZ0(config.z0));
      if (config.z1 != null) dispatch(objectiveSlice.setPosZ1(config.z1));
      if (config.pixelsize != null) dispatch(objectiveSlice.setPixelSize(config.pixelsize));
      if (config.magnification != null) dispatch(objectiveSlice.setMagnification(config.magnification));
      if (config.NA != null) dispatch(objectiveSlice.setNA(config.NA));
      if (config.objectiveName != null) dispatch(objectiveSlice.setObjectiveName(config.objectiveName));

      dispatch(objectiveSlice.setmagnification1(config.availableObjectiveMagnifications?.[0] ?? 0));
      dispatch(objectiveSlice.setmagnification2(config.availableObjectiveMagnifications?.[1] ?? 0));
      dispatch(objectiveSlice.setAvailableObjectivesNames(config.availableObjectivesNames || ["Obj 1", "Obj 2"]));
      dispatch(objectiveSlice.setAvailableObjectiveMagnifications(config.availableObjectiveMagnifications || [0, 0]));
      dispatch(objectiveSlice.setAvailableObjectiveNAs(config.availableObjectiveNAs || [0, 0]));
      dispatch(objectiveSlice.setAvailableObjectivePixelSizes(config.availableObjectivePixelSizes || [0, 0]));

      if (config.currentObjective != null) {
        dispatch(objectiveSlice.setCurrentObjective(config.currentObjective));
      }
    })
    .catch((err) => {
      console.error("Failed to fetch objective status", err);
    });
};

export default fetchObjectiveControllerGetStatus;
