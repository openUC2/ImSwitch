import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";

import apiGetCurrentExperimentParams from "../backendapi/apiExperimentControllerGetCurrentExperimentParams.js";

  

const fetchExperimentControllerGetCurrentExperimentParams = (dispatch) => {
    //request
  apiGetCurrentExperimentParams()
    .then((data) => {
      console.log("fetchExperimentControllerGetCurrentExperimentParams", data);
      //update redux states 
      dispatch(parameterRangeSlice.setIllumination(data.illuSources || []));
      dispatch(parameterRangeSlice.setIlluminationIntensities(data.illuIntensities || []));
      dispatch(parameterRangeSlice.setTimeLapsePeriodMin(data.timeLapsePeriodMin || 0));
      dispatch(parameterRangeSlice.setTimeLapsePeriodMax(data.timeLapsePeriodMax || 0));
      dispatch(parameterRangeSlice.setNumberOfImagesMin(data.numberOfImagesMin || 1));
      dispatch(parameterRangeSlice.setNumberOfImagesMax(data.numberOfImagesMax || 1));
      dispatch(parameterRangeSlice.setAutoFocusMin(data.autofocusMinFocusPosition || 0));
      dispatch(parameterRangeSlice.setAutoFocusMax(data.autofocusMaxFocusPosition || 0));
      dispatch(parameterRangeSlice.setAutoFocusStepSizeMin(data.autofocusStepSizeMin || 1));
      dispatch(parameterRangeSlice.setAutoFocusStepSizeMax(data.autofocusStepSizeMax || 1));
      dispatch(parameterRangeSlice.setZStackMin(data.zStackMinFocusPosition || 0));
      dispatch(parameterRangeSlice.setZStackMax(data.zStackMaxFocusPosition || 0));
      dispatch(parameterRangeSlice.setZStackStepSizeMin(data.zStackStepSizeMin || 1));
      dispatch(parameterRangeSlice.setZStackStepSizeMax(data.zStackStepSizeMax || 1));
      dispatch(parameterRangeSlice.setSpeed([1, 5, 10, 50, 100, 500, 1000, 10000, 20000, 100000]));

      // New code to handle additional data fields
      dispatch(parameterRangeSlice.setIlluSources(data.illuSources || []));
      dispatch(parameterRangeSlice.setIlluSourceMinIntensities(data.illuSourceMinIntensities || [0]));
      dispatch(parameterRangeSlice.setIlluSourceMaxIntensities(data.illuSourceMaxIntensities || [1023]));
      dispatch(parameterRangeSlice.setilluIntensities(data.illuIntensities || [0]));
      dispatch(parameterRangeSlice.setExposureTimes(data.exposureTimes || [0]));
      dispatch(parameterRangeSlice.setGains(data.gains || [0]));
      dispatch(parameterRangeSlice.setIsDPCpossible(data.isDPCpossible || false));
      dispatch(parameterRangeSlice.setIsDarkfieldpossible(data.isDarkfieldpossible || false));

    })
    .catch((err) => {
      console.error("Failed to fetch experiment parameters", err);
    });
};

export default fetchExperimentControllerGetCurrentExperimentParams;
