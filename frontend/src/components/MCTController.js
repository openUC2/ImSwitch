import React, { useContext, useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { MCTContext } from "../context/MCTContext";
import {
  Paper,
  Grid,
  TextField,
  Checkbox,
  FormControlLabel,
  Slider,
  Button,
  Typography,
} from "@mui/material";
import * as widgetSlice from "../state/slices/WidgetSlice.js";
//import { useWebSocket } from "../context/useWebSocket";


const MCTController = ({ hostIP, hostPort }) => {
  const [numImagesTaken, setNumImagesTaken] = useState(0);
  const [folderPath, setFolderPath] = useState("");
  //const socket = useWebSocket();
  const {
    timePeriod,
    setTimePeriod,
    numMeasurements,
    setNumMeasurements,
    zMin,
    setZMin,
    zMax,
    setZMax,
    zSteps,
    setZSteps,
    zStackEnabled,
    setZStackEnabled,
    xMin,
    setXMin,
    xMax,
    setXMax,
    xSteps,
    setXSteps,
    xStackEnabled,
    setXStackEnabled,
    yMin,
    setYMin,
    yMax,
    setYMax,
    ySteps,
    setYSteps,
    yStackEnabled,
    setYStackEnabled,
    intensityLaser1,
    setIntensityLaser1,
    intensityLaser2,
    setIntensityLaser2,
    intensityLED,
    setIntensityLED,
    fileName,
    setFileName,
    isRunning,
    setIsRunning,
  } = useContext(MCTContext);

  // Redux dispatcher and state
  const dispatch = useDispatch();
  const widgetState = useSelector(widgetSlice.getWidgetState);

  /*
  useEffect(() => {
    if (socket) {
      socket.onmessage = (event) => {
        console.log("Message received:", event.data);
        
      };
    }
    return () => {
      if (socket) {
        socket.onmessage = null;
      }
    };
  }, [socket]);

  */
  useEffect(() => {
    const fetchMCTStatus = () => {
      const url = `${hostIP}:${hostPort}/imswitch/api/MCTController/getMCTStatus`;

  
      fetch(url)
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
          // Set default values from the response
         // TODO: Should we fetch this on the first render?          
          
          setTimePeriod(data.timePeriod);
          setZStackEnabled(data.zStackEnabled);
          setZMin(data.zStackMin);
          setZMax(data.zStackMax);
          setZSteps(data.zStackStep);
          setXStackEnabled(data.xyScanEnabled);
          setXMin(data.xScanMin);
          setXMax(data.xScanMax);
          setXSteps(data.xScanStep);
          setYMin(data.yScanMin);
          setYMax(data.yScanMax);
          setYSteps(data.yScanStep);
          setIntensityLaser1(data.Illu1Value);
          setIntensityLaser2(data.Illu2Value);
          setIntensityLED(data.Illu3Value);
          
          // enable/disable start/stop
          setNumImagesTaken(data.nImagesTaken);
          setIsRunning(data.isMCTrunning);
          setFolderPath(data.MCTFilename);
        })
        .catch((error) => {
          //console.error("Error fetching MCT status:", error);
        });
      };
      
    fetchMCTStatus();

  }, [hostIP, hostPort]);
  

  
  const handleStart = () => {
    const url =
      `${hostIP}:${hostPort}/imswitch/api/MCTController/startTimelapseImaging?` +
      `tperiod=${timePeriod}&nImagesToCapture=${numMeasurements}&MCTFilename=${fileName}&` +
      `zStackEnabled=${zStackEnabled}&zStackMin=${zMin}&zStackMax=${zMax}&zStackStep=${zSteps}&` +
      `xyScanEnabled=${xStackEnabled}&xScanMin=${xMin}&xScanMax=${xMax}&xScanStep=${xSteps}&` +
      `yScanMin=${yMin}&yScanMax=${yMax}&yScanStep=${ySteps}&` +
      `IlluValue1=${intensityLaser1}&IlluValue2=${intensityLaser2}&IlluValue3=${intensityLED}`;

    fetch(url, { method: "GET" })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        setIsRunning(true);
      })
      .catch((error) => console.error("Error:", error));
  };

  const handleStop = () => {
    const url = `${hostIP}:${hostPort}/imswitch/api/MCTController/stopTimelapseImaging`;
    fetch(url, { method: "GET" })
      .then((response) => response.json())
      .then((data) => {
        console.log(data);
        setIsRunning(false);
      })
      .catch((error) => console.error("Error:", error));
  };

  return (
    <Paper style={{ padding: "20px" }}>
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <TextField
            label="Period T (s)"
            value={timePeriod}
            onChange={(e) => setTimePeriod(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="N Measurements"
            value={numMeasurements}
            onChange={(e) => setNumMeasurements(e.target.value)}
            fullWidth
          />
        </Grid>
        {/* Z-Stack, X-Stack, Y-Stack UI */}
        <Grid item xs={3}>
          <TextField
            label="Z-Stack Min"
            value={zMin}
            onChange={(e) => setZMin(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Z-Stack Max"
            value={zMax}
            onChange={(e) => setZMax(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Z-Stack Steps"
            value={zSteps}
            onChange={(e) => setZSteps(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <FormControlLabel
            control={<Checkbox />}
            checked={zStackEnabled}
            onChange={(e) => setZStackEnabled(e.target.checked)}
            label="Z-Stack Enabled"
          />
        </Grid>
        {/* XY Scan and Y Scan */}
        <Grid item xs={3}>
          <TextField
            label="X Scan Min"
            value={xMin}
            onChange={(e) => setXMin(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="X Scan Max"
            value={xMax}
            onChange={(e) => setXMax(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="X Scan Steps"
            value={xSteps}
            onChange={(e) => setXSteps(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <FormControlLabel
            control={<Checkbox />}
            checked={xStackEnabled}
            onChange={(e) => setXStackEnabled(e.target.checked)}
            label="XY Scan Enabled"
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Y Scan Min"
            value={yMin}
            onChange={(e) => setYMin(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Y Scan Max"
            value={yMax}
            onChange={(e) => setYMax(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <TextField
            label="Y Scan Steps"
            value={ySteps}
            onChange={(e) => setYSteps(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <FormControlLabel
            control={<Checkbox />}
            checked={yStackEnabled}
            onChange={(e) => setYStackEnabled(e.target.checked)}
            label="Y-Stack Enabled"
          />
        </Grid>
        {/* Intensity Controls */}
        <Grid item xs={12}>
          <Typography>Intensity (Laser 1): {intensityLaser1}</Typography>
          <Slider
            value={widgetState.sliderValue}
            onChange={(e, value) => dispatch(widgetSlice.setSliderValue(value))}
            max={32767}
            step={1}
          />
        </Grid>
        <Grid item xs={12}>
          <Typography>
            Intensity (Laser 2): {widgetState.generic["slider2"] || 0}
          </Typography>
          <Slider
            value={widgetState.generic["slider2"] || 0}
            onChange={(e, value) => dispatch(widgetSlice.updateGeneric({ key: "slider2", value }))}
            max={32767}
            step={1}
          />
        </Grid>
        <Grid item xs={12}>
          <Typography>
            Intensity (LED): {widgetState.generic["intensity"] || 0}
          </Typography>
          <Slider
            value={widgetState.generic["intensity"] || 0}
            onChange={(e, value) =>
              dispatch(widgetSlice.updateGeneric({ key: "intensity", value }))
            }
            max={255}
            step={1}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            label="File Name"
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            fullWidth
          />
        </Grid>
        <Grid item xs={3}>
          <Typography variant="body1" color="textSecondary">
            {`Images taken: ${numImagesTaken}`}
          </Typography>
        </Grid>
        <Grid item xs={3}>
          <Typography variant="body1" color="textSecondary">
            {`Folder: ${folderPath}`}
          </Typography>
        </Grid>
        <Grid item xs={12}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleStart}
            disabled={isRunning}
          >
            Start
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleStop}
            disabled={!isRunning}
            style={{ marginLeft: "10px" }}
          >
            Stop
          </Button>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default MCTController;
