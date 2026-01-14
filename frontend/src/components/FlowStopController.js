import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
  TextField,
  Button,
  Slider,
  Grid,
} from "@mui/material";
import { green, red } from "@mui/material/colors";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import { useWebSocket } from "../context/WebSocketContext";
import * as flowStopSlice from "../state/slices/FlowStopSlice.js";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

const FlowStopController = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  // Redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const flowStopState = useSelector(flowStopSlice.getFlowStopState);

  // Use Redux state instead of local useState
  const tabIndex = flowStopState.tabIndex;
  const timeStamp = flowStopState.timeStamp;
  const experimentName = flowStopState.experimentName;
  const experimentDescription = flowStopState.experimentDescription;
  const uniqueId = flowStopState.uniqueId;
  const numImages = flowStopState.numImages;
  const volumePerImage = flowStopState.volumePerImage;
  const timeToStabilize = flowStopState.timeToStabilize;
  const pumpSpeed = flowStopState.pumpSpeed;
  const isRunning = flowStopState.isRunning;
  const currentImageCount = flowStopState.currentImageCount;
  const socket = useWebSocket();

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/FlowStopController/getStatus`
        );
        const data = await response.json();
        dispatch(flowStopSlice.setIsRunning(data[0]));
        dispatch(flowStopSlice.setCurrentImageCount(data[1]));
      } catch (error) {
        //console.error('Error fetching status:', error);
      }
    };

    const fetchExperimentParameters = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/FlowStopController/getExperimentParameters`
        );
        const data = await response.json();
        dispatch(flowStopSlice.setTimeStamp(data.timeStamp));
        dispatch(flowStopSlice.setExperimentName(data.experimentName));
        dispatch(
          flowStopSlice.setExperimentDescription("Add some description here")
        );
        dispatch(flowStopSlice.setUniqueId(parseFloat(data.uniqueId, 1)));
        dispatch(flowStopSlice.setNumImages(parseFloat(data.numImages, -1)));
        dispatch(
          flowStopSlice.setVolumePerImage(parseFloat(data.volumePerImage, 1000))
        );
        dispatch(
          flowStopSlice.setTimeToStabilize(parseFloat(data.timeToStabilize, 1))
        );
        dispatch(flowStopSlice.setPumpSpeed(parseFloat(data.pumpSpeed, 1000)));
      } catch (error) {
        console.error("Error fetching experiment parameters:", error);
      }
    };

    fetchStatus();
    fetchExperimentParameters();
  }, [hostIP, hostPort]);

  // connect to the web
  useEffect(() => {
    if (!socket) return;

    socket.on("signal", (data) => {
      const jdata = JSON.parse(data);
      if (jdata.name === "sigImagesTaken") {
        dispatch(flowStopSlice.setCurrentImageCount(jdata.args.p0));
      }
      if (jdata.name === "sigIsRunning") {
        dispatch(flowStopSlice.setIsRunning(jdata.args.p0));
      }
    });
    // Clean up the chart on component unmount
    return () => {
      if (socket) {
        socket.onmessage = null;
      }
    };
  }, [socket]);

  const startExperiment = () => {
    // https://localhost/imswitch/api/FlowStopController/startFlowStopExperimentFastAPI?timeStamp=asdf&experimentName=adf&experimentDescription=asdf&uniqueId=asdf&numImages=19&volumePerImage=199&timeToStabilize=1&delayToStart=1&frameRate=1&filePath=.%2F&fileFormat=TIF&isRecordVideo=true&pumpSpeed=10000
    // Build URL from Redux connection settings
    const url = `${hostIP}:${hostPort}/imswitch/api/FlowStopController/startFlowStopExperimentFastAPI?timeStamp=${timeStamp}&experimentName=${experimentName}&experimentDescription=${experimentDescription}&uniqueId=${uniqueId}&numImages=${numImages}&volumePerImage=${volumePerImage}&timeToStabilize=${timeToStabilize}&isRecordVideo=true&pumpSpeed=${pumpSpeed}`;
    fetch(url, { method: "GET" })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        dispatch(flowStopSlice.setIsRunning(true));
      })
      .catch((error) => console.error("Error:", error));
  };

  const stopExperiment = () => {
    const url = `${hostIP}:${hostPort}/imswitch/api/FlowStopController/stopFlowStopExperiment`;

    fetch(url, { method: "GET" })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        dispatch(flowStopSlice.setIsRunning(false));
      })
      .catch((error) => console.error("Error:", error));
  };

  const handleTabChange = (event, newValue) => {
    dispatch(flowStopSlice.setTabIndex(newValue));
  };

  return (
    <Paper>
      <Tabs
        value={tabIndex}
        onChange={handleTabChange}
        aria-label="acquisition settings tabs"
      >
        <Tab label="Automatic Settings" />
        <Tab label="Manual Acquisition Settings" />
      </Tabs>

      <TabPanel value={tabIndex} index={1}>
        <Typography>Focus</Typography>
        <Slider defaultValue={30} />
        <Typography>Pump Speed</Typography>
        <Slider defaultValue={30} />
        <Button variant="contained">Snap</Button>
        <TextField label="Exposure Time" defaultValue="0.1" />
        <TextField label="Gain" defaultValue="0" />
      </TabPanel>

      <TabPanel value={tabIndex} index={0}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Time Stamp Name"
              value={timeStamp}
              onChange={(e) =>
                dispatch(flowStopSlice.setTimeStamp(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Experiment Name"
              value={experimentName}
              onChange={(e) =>
                dispatch(flowStopSlice.setExperimentName(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Experiment Description"
              value={experimentDescription}
              onChange={(e) =>
                dispatch(flowStopSlice.setExperimentDescription(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Volume Per Image"
              value={volumePerImage}
              onChange={(e) =>
                dispatch(flowStopSlice.setVolumePerImage(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Time to stabilize"
              value={timeToStabilize}
              onChange={(e) =>
                dispatch(flowStopSlice.setTimeToStabilize(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Pump Speed"
              value={pumpSpeed}
              onChange={(e) =>
                dispatch(flowStopSlice.setPumpSpeed(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              style={{ marginBottom: "20px" }}
              label="Number of Images"
              value={numImages}
              onChange={(e) =>
                dispatch(flowStopSlice.setNumImages(e.target.value))
              }
              fullWidth
            />
          </Grid>
          <Grid item xs={12}>
            <div>
              <Button
                style={{ marginBottom: "20px", marginRight: "10px" }}
                variant="contained"
                onClick={startExperiment}
                disabled={isRunning}
              >
                Start
              </Button>
              <Button
                style={{ marginBottom: "20px" }}
                variant="contained"
                onClick={stopExperiment}
                disabled={!isRunning}
              >
                Stop
              </Button>
            </div>
          </Grid>
          <Grid item xs={6}>
            <Box display="flex" alignItems="center">
              <Typography variant="h6">Status: </Typography>
              {isRunning ? (
                <CheckCircleIcon
                  style={{ color: green[500], marginLeft: "10px" }}
                />
              ) : (
                <CancelIcon style={{ color: red[500], marginLeft: "10px" }} />
              )}
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="h6">
              Images Taken: {currentImageCount}
            </Typography>
          </Grid>
        </Grid>
      </TabPanel>
    </Paper>
  );
};

export default FlowStopController;
