import apiObjectiveControllerGetCurrentObjective from "../backendapi/apiObjectiveControllerGetCurrentObjective.js";

import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";

const fetchObjectiveControllerGetCurrentObjective = (dispatch) => {
  // Request to fetch current objective 
  apiObjectiveControllerGetCurrentObjective()
    .then((data) => {
      console.log("fetchObjectiveControllerGetCurrentObjective", data);

      // Update Redux with the fetched data
      dispatch(objectiveSlice.setCurrentObjective(data[0])); 
      dispatch(objectiveSlice.setObjectiveName(data[1])); 
      /*reposne: 
        [
        1,
        "10x"
        ]
      */
    })
    .catch((err) => {
      console.error("Failed to fetch objective status", err);
    });
};

export default fetchObjectiveControllerGetCurrentObjective;
