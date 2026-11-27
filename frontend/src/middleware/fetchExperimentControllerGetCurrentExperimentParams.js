import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";

import apiGetCurrentExperimentParams from "../backendapi/apiExperimentControllerGetCurrentExperimentParams.js";
import apiLaserControllerGetLaserNames from "../backendapi/apiLaserControllerGetLaserNames.js";
import apiLaserControllerGetLaserValueRanges from "../backendapi/apiLaserControllerGetLaserValueRanges.js";

  

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
      dispatch(parameterRangeSlice.setScanSpeedXY(data.scanSpeedXY || 20000));

      // New code to handle additional data fields
      dispatch(parameterRangeSlice.setIlluSources(data.illuSources || []));
      // Per-source kind tag (parallel to illuSources): "default" | "ring" | "dpc".
      // Backends predating the LEDMatrix synthetic channels omit this field —
      // the slice falls back to empty (all-default) which the UI tolerates.
      dispatch(parameterRangeSlice.setIlluSourceKinds(data.illuSourceKinds || []));
      dispatch(parameterRangeSlice.setSyntheticChannels(data.syntheticChannels || []));
      // LED matrix hardware metadata (nLedsX/nLedsY/maxRingRadius) — null
      // when no LED matrix is configured.  Drives the ring-radius slider
      // bounds in the Wellplate designer.
      dispatch(parameterRangeSlice.setLedMatrixInfo(data.ledMatrixInfo || null));
      dispatch(parameterRangeSlice.setIlluSourceMinIntensities(data.illuSourceMinIntensities || [0]));
      dispatch(parameterRangeSlice.setIlluSourceMaxIntensities(data.illuSourceMaxIntensities || [1023]));
      dispatch(parameterRangeSlice.setilluIntensities(data.illuIntensities || [0]));
      dispatch(parameterRangeSlice.setExposureTimes(data.exposureTimes || [0]));
      dispatch(parameterRangeSlice.setGains(data.gains || [0]));
      dispatch(parameterRangeSlice.setIsDPCpossible(data.isDPCpossible || false));
      dispatch(parameterRangeSlice.setIsDarkfieldpossible(data.isDarkfieldpossible || false));

    })
    .catch(async (err) => {
      // Setups without the Experiment widget (e.g. a lightsheet) have no
      // ExperimentController: take the channels from the LaserController instead,
      // otherwise the UI keeps the persisted list of the previous setup.
      console.warn("No experiment parameters, using LaserController for the channel list", err);
      try {
        const names = await apiLaserControllerGetLaserNames();
        const ranges = await Promise.all(
          names.map((n) => apiLaserControllerGetLaserValueRanges(n).catch(() => [0, 1023])),
        );
        dispatch(parameterRangeSlice.setIllumination(names));
        dispatch(parameterRangeSlice.setIlluSources(names));
        dispatch(parameterRangeSlice.setIlluSourceKinds(names.map(() => "default")));
        dispatch(parameterRangeSlice.setSyntheticChannels([]));
        dispatch(parameterRangeSlice.setLedMatrixInfo(null));
        dispatch(parameterRangeSlice.setIlluSourceMinIntensities(ranges.map((r) => r?.[0] ?? 0)));
        dispatch(parameterRangeSlice.setIlluSourceMaxIntensities(ranges.map((r) => r?.[1] ?? 1023)));
        dispatch(parameterRangeSlice.setilluIntensities(names.map(() => 0)));
      } catch (laserErr) {
        console.error("Failed to fetch the laser channels", laserErr);
      }
    });
};

export default fetchExperimentControllerGetCurrentExperimentParams;
