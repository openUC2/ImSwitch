import React, { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Box,
  Chip,
  Stack,
  Divider,
  Paper,
  Tooltip,
  IconButton,
} from "@mui/material";
import {
  Add,
  Remove,
  Home,
  Stop,
  Speed,
  Edit,
  Check,
  Close,
} from "@mui/icons-material";
import * as positionSlice from "../state/slices/PositionSlice.js";
import { useSelector, useDispatch } from "react-redux";

/**
 * ImprovedAxisControl - Consolidated position display with multi-purpose editing
 */
const ImprovedAxisControl = ({ hostIP, hostPort }) => {
  const dispatch = useDispatch();
  const [positionerName, setPositionerName] = useState("");
  const [globalSpeed, setGlobalSpeed] = useState(20000);
  const [editingAxis, setEditingAxis] = useState(null); // Which axis position is being edited
  const [tempPositions, setTempPositions] = useState({}); // Temporary values during editing

  // Get positions and step sizes from Redux
  const positionState = useSelector(positionSlice.getPositionState);

  // Map Redux state to positions object (x, y, z, a -> X, Y, Z, A)
  const positions = {
    X: positionState.x,
    Y: positionState.y,
    Z: positionState.z,
    A: positionState.a,
  };

  // Get step sizes from Redux (persisted)
  const stepSizes = positionState.stepSizes || {
    X: 100,
    Y: 100,
    Z: 10,
    A: 10,
  };

  /* --- initial fetch for positioner name --- */
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/PositionerController/getPositionerNames`
        );
        const d = await r.json();
        setPositionerName(d[0]);
      } catch (e) {
        console.error(e);
      }
    })();
  }, [hostIP, hostPort]);

  // Quick step sizes options
  const stepOptions = [
    { value: 1, label: "1µm", color: "info" },
    { value: 10, label: "10µm", color: "primary" },
    { value: 100, label: "100µm", color: "secondary" },
    { value: 1000, label: "1000µm", color: "success" },
    { value: 10000, label: "10000µm", color: "warning" },
  ];

  // Update step size for a specific axis (persisted in Redux)
  const setStepSize = (axis, value) => {
    dispatch(positionSlice.setStepSize({ axis, value }));
  };

  const call = async (url) => {
    try {
      await fetch(url);
    } catch (e) {
      console.error(e);
    }
  };

  const base = `${hostIP}:${hostPort}/imswitch/api/PositionerController`;

  const moveAxis = (axis, distance) =>
    call(
      `${base}/movePositioner?positionerName=${positionerName}` +
        `&axis=${axis}&dist=${distance}&isAbsolute=false&isBlocking=false&speed=${globalSpeed}`
    );

  const moveAxisAbs = (axis, target) =>
    call(
      `${base}/movePositioner?positionerName=${positionerName}` +
        `&axis=${axis}&dist=${target}&isAbsolute=true&isBlocking=false&speed=${globalSpeed}`
    );

  const homeAxis = (axis) =>
    call(
      `${base}/homeAxis?positionerName=${positionerName}` +
        `&axis=${axis}&isBlocking=false`
    );

  const stopAxis = (axis) =>
    call(`${base}/stopAxis?positionerName=${positionerName}&axis=${axis}`);

  const homeAll = () => {
    Object.keys(positions).forEach((axis) => homeAxis(axis));
  };

  const stopAll = () => {
    Object.keys(positions).forEach((axis) => stopAxis(axis));
  };

  // Position editing functions
  const startEditing = (axis) => {
    setEditingAxis(axis);
    setTempPositions({ ...tempPositions, [axis]: positions[axis] || 0 });
  };

  const cancelEditing = () => {
    setEditingAxis(null);
    setTempPositions({});
  };

  const confirmMove = () => {
    if (editingAxis && tempPositions[editingAxis] !== undefined) {
      moveAxisAbs(editingAxis, tempPositions[editingAxis]);
      setEditingAxis(null);
      setTempPositions({});
    }
  };

  const updateTempPosition = (axis, value) => {
    setTempPositions({ ...tempPositions, [axis]: value });
  };

  return (
    <Card sx={{ minWidth: 300 }}>
      <CardContent>
        {/* Consolidated Position Display with Multi-Purpose Editing */}
        <Paper
          sx={{
            p: 2,
            mb: 2,
            bgcolor: (theme) =>
              theme.palette.mode === "dark" ? "grey.800" : "grey.100",
            color: (theme) =>
              theme.palette.mode === "dark" ? "common.white" : "text.primary",
          }}
        >
          <Typography
            variant="subtitle2"
            sx={{
              mb: 1,
              color: (theme) =>
                theme.palette.mode === "dark"
                  ? "success.light"
                  : "success.dark",
            }}
          >
            POSITION CONTROL (µm) - Click to Edit Target
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(positions).map(([axis, position]) => (
              <Grid item xs={3} key={axis}>
                <Box sx={{ textAlign: "center" }}>
                  <Typography
                    variant="caption"
                    sx={{
                      color: (theme) =>
                        theme.palette.mode === "dark"
                          ? "grey.400"
                          : "text.secondary",
                      mb: 1,
                      display: "block",
                    }}
                  >
                    {axis}-AXIS
                  </Typography>

                  {editingAxis === axis ? (
                    // Edit Mode
                    <Box
                      sx={{ display: "flex", flexDirection: "column", gap: 1 }}
                    >
                      <TextField
                        size="small"
                        type="number"
                        value={tempPositions[axis] || 0}
                        onChange={(e) =>
                          updateTempPosition(axis, Number(e.target.value))
                        }
                        sx={{
                          "& .MuiInputBase-root": {
                            color: "white",
                            fontFamily: "monospace",
                            fontSize: "0.9rem",
                          },
                          "& .MuiOutlinedInput-notchedOutline": {
                            borderColor: "success.light",
                          },
                        }}
                      />
                      <Box
                        sx={{
                          display: "flex",
                          gap: 0.5,
                          justifyContent: "center",
                        }}
                      >
                        <IconButton
                          size="small"
                          onClick={confirmMove}
                          sx={{ color: "success.light" }}
                        >
                          <Check fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={cancelEditing}
                          sx={{ color: "error.light" }}
                        >
                          <Close fontSize="small" />
                        </IconButton>
                      </Box>
                    </Box>
                  ) : (
                    // Display Mode
                    <Box
                      sx={{ cursor: "pointer" }}
                      onClick={() => startEditing(axis)}
                    >
                      <Typography
                        variant="h6"
                        sx={{
                          fontFamily: "monospace",
                          color: "success.light",
                          fontWeight: "bold",
                          "&:hover": { color: "warning.light" },
                        }}
                      >
                        {(position || 0).toFixed(1)}
                      </Typography>
                      <Edit
                        fontSize="small"
                        sx={{ color: "grey.500", mt: 0.5 }}
                      />
                    </Box>
                  )}
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>

        <Grid container spacing={1.5}>
          {Object.keys(positions).map((axis) => (
            <Grid item xs={6} sm={6} md={6} lg={6} xl={3} key={axis}>
              <Paper
                sx={{
                  p: 1.5,
                  bgcolor: "background.paper",
                  border: 1,
                  borderColor: "divider",
                }}
              >
                <Typography
                  variant="subtitle2"
                  sx={{
                    textAlign: "center",
                    fontWeight: "bold",
                    mb: 1,
                    color: "primary.main",
                  }}
                >
                  {axis} Axis
                </Typography>

                {/* Relative Movement */}
                <Stack direction="row" spacing={0.5} sx={{ mb: 1 }}>
                  <Tooltip title={`-${stepSizes[axis]}µm`}>
                    <IconButton
                      color="error"
                      size="small"
                      onClick={() => moveAxis(axis, -stepSizes[axis])}
                      sx={{
                        flex: 1,
                        border: 1,
                        borderColor: "error.main",
                        borderRadius: 1,
                        "&:hover": { bgcolor: "error.light" },
                      }}
                    >
                      <Remove fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={`+${stepSizes[axis]}µm`}>
                    <IconButton
                      color="success"
                      size="small"
                      onClick={() => moveAxis(axis, +stepSizes[axis])}
                      sx={{
                        flex: 1,
                        border: 1,
                        borderColor: "success.main",
                        borderRadius: 1,
                        "&:hover": { bgcolor: "success.light" },
                      }}
                    >
                      <Add fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Stack>

                {/* Control Buttons */}
                <Stack direction="column" spacing={0.5} sx={{ mb: 1 }}>
                  <Tooltip title="Home axis">
                    <Button
                      variant="outlined"
                      color="secondary"
                      size="small"
                      onClick={() => homeAxis(axis)}
                      sx={{ fontSize: "0.7rem", py: 0.5, minHeight: 28 }}
                      startIcon={<Home sx={{ fontSize: "1rem" }} />}
                    >
                      HOME
                    </Button>
                  </Tooltip>
                  <Tooltip title="Stop axis">
                    <Button
                      variant="outlined"
                      color="error"
                      size="small"
                      onClick={() => stopAxis(axis)}
                      sx={{ fontSize: "0.7rem", py: 0.5, minHeight: 28 }}
                      startIcon={<Stop sx={{ fontSize: "1rem" }} />}
                    >
                      STOP
                    </Button>
                  </Tooltip>
                </Stack>

                {/* Step Size Selection */}
                <Box>
                  <Typography
                    variant="caption"
                    sx={{ mb: 0.5, display: "block", fontSize: "0.7rem" }}
                  >
                    Step Size:
                  </Typography>
                  <Stack
                    direction="row"
                    spacing={0.5}
                    sx={{ flexWrap: "wrap", gap: 0.5 }}
                  >
                    {stepOptions.map((option) => (
                      <Chip
                        key={option.value}
                        label={option.label}
                        size="small"
                        color={
                          stepSizes[axis] === option.value
                            ? option.color
                            : "default"
                        }
                        variant={
                          stepSizes[axis] === option.value
                            ? "filled"
                            : "outlined"
                        }
                        onClick={() => setStepSize(axis, option.value)}
                        sx={{
                          cursor: "pointer",
                          fontSize: "0.65rem",
                          height: 22,
                        }}
                      />
                    ))}
                  </Stack>
                  <TextField
                    size="small"
                    type="number"
                    value={stepSizes[axis]}
                    onChange={(e) => setStepSize(axis, Number(e.target.value))}
                    sx={{ width: "100%", mt: 0.5 }}
                    InputProps={{
                      style: { fontSize: "0.75rem", height: 32 },
                      endAdornment: (
                        <Typography variant="caption">µm</Typography>
                      ),
                    }}
                  />
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>

        <Divider sx={{ my: 2 }} />

        {/* Global Controls */}
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 2,
          }}
        >
          <TextField
            label="Global Speed"
            type="number"
            size="small"
            value={globalSpeed}
            onChange={(e) => setGlobalSpeed(Number(e.target.value))}
            sx={{ width: 150 }}
            InputProps={{
              startAdornment: <Speed fontSize="small" sx={{ mr: 0.5 }} />,
              style: { fontSize: "0.8rem" },
            }}
          />

          <Stack direction="row" spacing={1}>
            <Button
              variant="contained"
              color="secondary"
              onClick={homeAll}
              startIcon={<Home />}
              size="small"
            >
              HOME ALL
            </Button>
            <Button
              variant="contained"
              color="error"
              onClick={stopAll}
              startIcon={<Stop />}
              size="small"
            >
              STOP ALL
            </Button>
          </Stack>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ImprovedAxisControl;
