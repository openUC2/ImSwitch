import React, { useState, useEffect, useRef } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../backendapi/createAxiosInstance';

import { Button, ButtonGroup, LinearProgress } from "@mui/material";

import * as wsUtils from "./WellSelectorUtils.js";
import * as coordinateCalculator from "./CoordinateCalculator.js";

import InfoPopup from "./InfoPopup.js";

import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as experimentStatusSlice from "../state/slices/ExperimentStatusSlice.js";
import * as experimentStateSlice from "../state/slices/ExperimentStateSlice.js";
import * as wellSelectorSlice from "../state/slices/WellSelectorSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";

import apiExperimentControllerStartWellplateExperiment from "../backendapi/apiExperimentControllerStartWellplateExperiment.js";
import apiExperimentControllerStopExperiment from "../backendapi/apiExperimentControllerStopExperiment.js";
import apiExperimentControllerPauseWorkflow from "../backendapi/apiExperimentControllerPauseWorkflow.js";
import apiExperimentControllerResumeExperiment from "../backendapi/apiExperimentControllerResumeExperiment.js";

import fetchGetExperimentStatus from "../middleware/fetchExperimentControllerGetExperimentStatus.js";
import { Shape } from "./WellSelectorCanvas.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import * as vizarrViewerSlice from "../state/slices/VizarrViewerSlice.js";
//##################################################################################
// Enum-like object for status
const Status = Object.freeze({
  IDLE: "idle",
  RUNNING: "running",
  PAUSED: "paused",
  STOPPING: "stopping",
});

//##################################################################################
const ExperimentComponent = () => {
  //state
  const infoPopupRef = useRef(null);

  //redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );
  // local state to enalbe open in VTK Viewer
  const [enableViewer, setEnableViewer] = useState(false);

  // helper for the workflow step state udpates
  const [cachedStepId, setCachedStepId] = useState(0);
  const [cachedTotalSteps, setCachedTotalSteps] = useState(undefined);
  const [cachedStepName, setCachedStepName] = useState("");

  // Access global Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const experimentWorkflowState = useSelector(
    experimentStateSlice.getExperimentState
  );

  const experimentStatusState = useSelector(
    experimentStatusSlice.getExperimentStatusState
  );
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);

  //##################################################################################
  useEffect(() => {
    //periodic experiment status fetch
    // Initial fetch
    fetchGetExperimentStatus(dispatch);

    // Set an interval to call fetchData every x seconds
    const intervalId = setInterval(() => {
      fetchGetExperimentStatus(dispatch);
    }, 3000); //Request period

    // Cleanup on component unmount
    return () => {
      clearInterval(intervalId);
    };
  }, []); // Empty dependency array ensures this runs once on mount

  //##################################################################################
  const handleStart = () => {
    console.log("Experiment started");

    // Update is_snakescan from wellSelectorState before creating the request
    dispatch(experimentSlice.setIsSnakescan(wellSelectorState.areaSelectSnakescan));
    
    // Also update overlap values from area select settings if in area select mode
    if (wellSelectorState.mode === 'area') {
      dispatch(experimentSlice.setOverlapWidth(wellSelectorState.areaSelectOverlap));
      dispatch(experimentSlice.setOverlapHeight(wellSelectorState.areaSelectOverlap));
    }

    // Use the new coordinate calculator to generate all scan coordinates
    console.log("Calculating scan coordinates in frontend...");
    const scanConfig = coordinateCalculator.calculateScanCoordinates(
      experimentState,
      objectiveState,
      wellSelectorState
    );
    
    console.log("Scan configuration:", scanConfig);
    console.log(`Total scan areas: ${scanConfig.scanAreas.length}`);
    console.log(`Total positions: ${scanConfig.metadata.totalPositions}`);

    // Zero out intensities for channels that are not enabled for experiment acquisition
    const channelEnabled = experimentState.parameterValue.channelEnabledForExperiment || [];
    const rawIntensities = experimentState.parameterValue.illuIntensities || [];
    const filteredIntensities = rawIntensities.map((val, idx) =>
      channelEnabled[idx] === true ? val : 0
      );

    //create experiment request with pre-calculated coordinates
    const experimentRequest = {
      name: experimentState.name,
      parameterValue: {
        ...experimentState.parameterValue,
        illuIntensities: filteredIntensities,
        resortPointListToSnakeCoordinates: false, // IMPORTANT: Tell backend NOT to resort
        is_snakescan: wellSelectorState.areaSelectSnakescan,
        overlapWidth: wellSelectorState.mode === 'area' ? wellSelectorState.areaSelectOverlap : experimentState.parameterValue.overlapWidth,
        overlapHeight: wellSelectorState.mode === 'area' ? wellSelectorState.areaSelectOverlap : experimentState.parameterValue.overlapHeight,
      },
      // Include scan configuration for backend to use
      scanAreas: scanConfig.scanAreas,
      scanMetadata: scanConfig.metadata,
      // Convert to backward-compatible point list format
      pointList: coordinateCalculator.convertToBackendFormat(scanConfig, experimentState).pointList,
    };

    // Convert object to JSON
    //const jsonString = JSON.stringify(experimentRequest);

    //show reuslt
    //console.log("experiment(obj):", experimentRequest);
    //console.log("experiment(json):", jsonString);

    //create request
    apiExperimentControllerStartWellplateExperiment(experimentRequest)
      .then((data) => {
        // Handle success response
        console.log("apiStartWellplateExperiment", data);
        //set state
        dispatch(experimentStatusSlice.setStatus(Status.RUNNING));
        // enable VTK Viewer
        setEnableViewer(true);
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage(
            "Experimental status has been updated to " +
              experimentWorkflowState.status
          );
        }
      })
      .catch((err) => {
        // Handle error
        //console.error("handleStart", err)
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Start Experiment failed");
        }
      });
  };

  const handleOpenVTKViewer = () => {
    console.log("Fetching last OME-Zarr path to open in VTK viewer");

    const api = createAxiosInstance();
    api.get(
      `/ExperimentController/getLastScanAsOMEZARR`
    )
      .then((res) => res.data)
      .then((data) => {
        // English comment: 'data' should contain the relative path like "/recordings/...ome.zarr"
        const lastZarrPath = data || "";
        // English comment: build the final URL for VTK viewer
        // Determine protocol based on current page or use http as default for local development
        const protocol = window.location.protocol === 'https:' ? 'https' : 'http';
        const viewerURL = `https://hms-dbmi.github.io/vizarr/?source=${protocol}://${connectionSettingsState.ip}:${connectionSettingsState.apiPort}/imswitch/data${lastZarrPath}`;
        console.log("Opening Vizarr with URL:", viewerURL);
        window.open(viewerURL, "_blank");
      })
      .catch((err) => {
        console.error("Failed to fetch the last OME-Zarr path:", err);
        infoPopupRef.current.showMessage(
          "Failed to open last OME-Zarr in viewer"
        );
      });
  };

  // Handler to open the integrated (offline) Vizarr viewer
  const handleOpenOfflineVizarr = () => {
    console.log("Fetching last OME-Zarr path to open in integrated Vizarr viewer");
    
    const api = createAxiosInstance();
    api.get(
      `/ExperimentController/getLastScanAsOMEZARR`
    )
      .then((res) => res.data)
      .then((data) => {
        // English comment: 'data' should contain the relative path like "/recordings/...ome.zarr"
        const lastZarrPath = data || "";
        if (lastZarrPath) {
          // Open the integrated Vizarr viewer with the path
          dispatch(vizarrViewerSlice.openViewer({
            url: lastZarrPath,
            fileName: lastZarrPath.split("/").pop() || "OME-Zarr"
          }));
          
          if (infoPopupRef.current) {
            infoPopupRef.current.showMessage("Opening OME-Zarr in integrated viewer");
          }
        } else {
          if (infoPopupRef.current) {
            infoPopupRef.current.showMessage("No OME-Zarr data available");
          }
        }
      })
      .catch((err) => {
        console.error("Failed to fetch the last OME-Zarr path:", err);
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Failed to open OME-Zarr in viewer");
        }
      });
  };
  //##################################################################################
  const handlePause = () => {
    console.log("Experiment paused");
    //create request
    apiExperimentControllerPauseWorkflow()
      .then((data) => {
        // Handle success response
        //set state
        dispatch(experimentStatusSlice.setStatus(Status.PAUSED));
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Experiment paused");
        }
      })
      .catch((err) => {
        // Handle error
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Pause Experiment failed");
        }
      });
  };

  //##################################################################################
  const handleResume = () => {
    console.log("Experiment resumed");
    //create request
    apiExperimentControllerResumeExperiment()
      .then((data) => {
        // Handle success response
        //set state
        dispatch(experimentStatusSlice.setStatus(Status.RUNNING));
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Experiment resumed...");
        }
      })
      .catch((err) => {
        // Handle error
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Resume Experiment failed");
        }
      });
  };

  //##################################################################################
  const handleStop = () => {
    console.log("Experiment stopped");
    //create request
    dispatch(experimentStatusSlice.setStatus(Status.IDLE)); // FIXME: This is a workaround to avoid the "stopping" status
    apiExperimentControllerStopExperiment()
      .then((data) => {
        // Handle success response
        //set state
        //set popup
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Experiment stopped");
        }
      })
      .catch((err) => {
        // Handle error
        //set popup
        console.log("handleStop", err);
        if (infoPopupRef.current) {
          infoPopupRef.current.showMessage("Stop Experiment failed");
        }
      });
  };

  //##################################################################################
  const showStartButton = () => {
    return (
      experimentStatusState.status === Status.IDLE ||
      experimentStatusState.status === Status.STOPPING
    );
  };

  //##################################################################################
  const showPauseButton = () => {
    return experimentStatusState.status === Status.RUNNING;
  };

  //##################################################################################
  const showResumeButton = () => {
    return experimentStatusState.status === Status.PAUSED;
  };

  //##################################################################################
  const showStopButton = () => {
    return experimentStatusState.status !== Status.IDLE;
  };

  useEffect(() => {
    if (experimentWorkflowState.totalSteps !== undefined) {
      setCachedTotalSteps(experimentWorkflowState.totalSteps);
    }
    if (experimentWorkflowState.stepId !== undefined) {
      setCachedStepId(experimentWorkflowState.stepId);
    }
    if (experimentWorkflowState.stepName) {
      setCachedStepName(experimentWorkflowState.stepName);
    }
  }, [
    experimentWorkflowState.totalSteps,
    experimentWorkflowState.stepId,
    experimentWorkflowState.stepName,
  ]);

  // Calculate progress percentage if we have valid totalSteps
  const progress =
    cachedTotalSteps && cachedTotalSteps > 0
      ? Math.floor((cachedStepId / cachedTotalSteps) * 100)
      : 0;

  //##################################################################################
  return (
    <div
      style={{
        textAlign: "left",
        padding: "10px",
        margin: "0px",
      }}
    >
      {/* Header */}
      <h4 style={{ margin: "0", padding: "0" }}>
        Experiment ({experimentStatusState.status})
      </h4>

      <ButtonGroup>
        <Button
          variant="contained"
          onClick={handleStart}
          disabled={!showStartButton()}
        >
          Start
        </Button>

        <Button
          variant="contained"
          onClick={handlePause}
          disabled={!showPauseButton()}
        >
          Pause
        </Button>

        <Button
          variant="contained"
          onClick={handleResume}
          disabled={!showResumeButton()}
        >
          Resume
        </Button>

        <Button
          variant="contained"
          onClick={handleStop}
          disabled={!showStopButton()}
        >
          Stop
        </Button>

        <Button
          variant="contained"
          onClick={handleOpenOfflineVizarr}
          title="Open OME-Zarr in integrated viewer (works offline)"
        >
          Open Vizarr (offline)
        </Button>

        <Button
          variant="outlined"
          onClick={handleOpenVTKViewer}
          title="Open OME-Zarr in external vizarr.io viewer (requires internet)"
        >
          Open External Vizarr
        </Button>

        {/* Display the step name (fixed width) and loading bar with percentage */}
        <span
          style={{
            marginLeft: 10,
            display: "inline-flex",
            alignItems: "center",
          }}
        >
          <div
            style={{
              width: "140px", // Fixed width for the step name
              overflow: "hidden", // Hide overflow if name is longer
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              marginRight: "8px",
            }}
            title={cachedStepName}
          >
            {cachedStepName}
          </div>

          {/* Only show loading bar if totalSteps is present */}
          {cachedTotalSteps && cachedTotalSteps > 0 && (
            <>
              <div style={{ width: "100px", marginRight: "8px" }}>
                <LinearProgress variant="determinate" value={progress} />
              </div>
              <span>{progress}%</span>
            </>
          )}
        </span>
      </ButtonGroup>

      {/* Header 
      <Snackbar
        open={popupOpen}
        autoHideDuration={null} // automatically hide
        //onClose={() => setPopupOpen(false)}
        message={popupMessage}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right", // You can change this to top, left, etc.
        }}
      />*/}

      <InfoPopup ref={infoPopupRef} />
    </div>
  );
};
//##################################################################################

export default ExperimentComponent;
