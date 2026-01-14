// ./components/ExtendedLEDMatrixController.js
import {
  Button,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import { useState } from "react";
import { useDispatch, useSelector } from "react-redux";

import illumination_circle from "../assets/illumination_circle.png";
import illumination_halves from "../assets/illumination_halves.png";
import illumination_ring from "../assets/illumination_ring.png";

import {
  getLEDMatrixState,
  setCircleRadius,
  setDirection,
  setIntensity,
  setIsOn,
  setMode,
  setRingRadius,
} from "../state/slices/LEDMatrixSlice";

import apiLEDMatrixControllerSetAllLED from "../backendapi/apiLEDMatrixControllerSetAllLED";
import apiLEDMatrixControllerSetCircle from "../backendapi/apiLEDMatrixControllerSetCircle";
import apiLEDMatrixControllerSetHalves from "../backendapi/apiLEDMatrixControllerSetHalves";
import apiLEDMatrixControllerSetRing from "../backendapi/apiLEDMatrixControllerSetRing";

const ExtendedLEDMatrixController = () => {
  const dispatch = useDispatch();
  const LEDMatrixState = useSelector(getLEDMatrixState);

  const [activeTab, setActiveTab] = useState(0);
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Local states for user inputs
  const [halvesIntensity, setHalvesIntensity] = useState(
    LEDMatrixState.intensity
  );
  const [halvesDirection, setHalvesDirection] = useState(
    LEDMatrixState.direction
  );

  const [ringRadius, setLocalRingRadius] = useState(LEDMatrixState.ringRadius);
  const [ringIntensity, setRingIntensity] = useState(LEDMatrixState.intensity);

  const [circleRadius, setLocalCircleRadius] = useState(
    LEDMatrixState.circleRadius
  );
  const [circleIntensity, setCircleIntensity] = useState(
    LEDMatrixState.intensity
  );

  const [allState, setAllState] = useState(LEDMatrixState.isOn ? 1 : 0);
  const [allIntensity, setAllIntensity] = useState(LEDMatrixState.intensity);

  // Button handlers
  const handleSetHalves = () => {
    dispatch(setMode("halves"));
    dispatch(setIntensity(halvesIntensity));
    dispatch(setDirection(halvesDirection));

    apiLEDMatrixControllerSetHalves({
      intensity: halvesIntensity,
      direction: halvesDirection,
    })
      .then((data) => {
        console.log("Halves set", data);
      })
      .catch((error) => {
        console.error("Error setting halves:", error);
      });
  };

  const handleSetRing = () => {
    dispatch(setMode("ring"));
    dispatch(setIntensity(ringIntensity));
    dispatch(setRingRadius(ringRadius));

    apiLEDMatrixControllerSetRing({
      ringRadius: ringRadius,
      intensity: ringIntensity,
    })
      .then((data) => {
        console.log("Ring set", data);
      })
      .catch((error) => {
        console.error("Error setting ring:", error);
      });
  };

  const handleSetCircle = () => {
    dispatch(setMode("circle"));
    dispatch(setIntensity(circleIntensity));
    dispatch(setCircleRadius(circleRadius));

    apiLEDMatrixControllerSetCircle({
      circleRadius: circleRadius,
      intensity: circleIntensity,
    })
      .then((data) => {
        console.log("Circle set", data);
      })
      .catch((error) => {
        console.error("Error setting circle:", error);
      });
  };

  const handleSetAll = () => {
    dispatch(setMode("all"));
    dispatch(setIsOn(allState === 1));
    dispatch(setIntensity(allIntensity));

    apiLEDMatrixControllerSetAllLED({
      state: allState,
      intensity: allIntensity,
      getReturn: true,
    })
      .then((data) => {
        console.log("All LED set", data);
      })
      .catch((error) => {
        console.error("Error setting all LED:", error);
      });
  };

  return (
    <Paper style={{ padding: "20px", marginTop: "20px" }}>
      <Typography variant="h6">LED Matrix Controller</Typography>
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        style={{ marginBottom: "20px" }}
      >
        <Tab label="Halves" />
        <Tab label="Ring" />
        <Tab label="Circle" />
        <Tab label="All" />
      </Tabs>

      {activeTab === 0 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src={illumination_halves}
              alt="Halves"
              style={{ maxHeight: "200px" }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="Intensity"
              type="number"
              value={halvesIntensity}
              onChange={(e) => setHalvesIntensity(Number(e.target.value))}
              fullWidth
            />
          </Grid>
          <Grid item xs={4}>
            <FormControl fullWidth>
              <InputLabel>Direction</InputLabel>
              <Select
                value={halvesDirection}
                label="Direction"
                onChange={(e) => setHalvesDirection(e.target.value)}
              >
                <MenuItem value="top">Top</MenuItem>
                <MenuItem value="bottom">Bottom</MenuItem>
                <MenuItem value="left">Left</MenuItem>
                <MenuItem value="right">Right</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={4}>
            <Button variant="contained" onClick={handleSetHalves} fullWidth>
              Set Halves
            </Button>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src= {`${process.env.PUBLIC_URL}/assets/illumination_ring.png`} /*{illumination_ring} */
              alt="Ring"
              style={{ maxHeight: "200px" }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="Ring Radius"
              type="number"
              value={ringRadius}
              onChange={(e) => setLocalRingRadius(Number(e.target.value))}
              fullWidth
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="Intensity"
              type="number"
              value={ringIntensity}
              onChange={(e) => setRingIntensity(Number(e.target.value))}
              fullWidth
            />
          </Grid>
          <Grid item xs={4}>
            <Button variant="contained" onClick={handleSetRing} fullWidth>
              Set Ring
            </Button>
          </Grid>
        </Grid>
      )}

      {activeTab === 2 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src= {`${process.env.PUBLIC_URL}/assets/illumination_circle.png`} /*{illumination_circle} */
              alt="Circle"
              style={{ maxHeight: "200px" }}
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="Circle Radius"
              type="number"
              value={circleRadius}
              onChange={(e) => setLocalCircleRadius(Number(e.target.value))}
              fullWidth
            />
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="Intensity"
              type="number"
              value={circleIntensity}
              onChange={(e) => setCircleIntensity(Number(e.target.value))}
              fullWidth
            />
          </Grid>
          <Grid item xs={4}>
            <Button variant="contained" onClick={handleSetCircle} fullWidth>
              Set Circle
            </Button>
          </Grid>
        </Grid>
      )}

      {activeTab === 3 && (
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <img
              src={`${process.env.PUBLIC_URL}/assets/illumination_all.png`}
              alt="All"
              style={{ maxHeight: "200px" }}
            />
          </Grid>
          <Grid item xs={4}>
            <FormControl fullWidth>
              <InputLabel>State</InputLabel>
              <Select
                value={allState}
                label="State"
                onChange={(e) => setAllState(e.target.value)}
              >
                <MenuItem value={1}>On</MenuItem>
                <MenuItem value={0}>Off</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={4}>
            <TextField
              label="Intensity"
              type="number"
              value={allIntensity}
              onChange={(e) => setAllIntensity(Number(e.target.value))}
              fullWidth
            />
          </Grid>
          <Grid item xs={4}>
            <Button variant="contained" onClick={handleSetAll} fullWidth>
              Set All
            </Button>
          </Grid>
        </Grid>
      )}
    </Paper>
  );
};

export default ExtendedLEDMatrixController;
