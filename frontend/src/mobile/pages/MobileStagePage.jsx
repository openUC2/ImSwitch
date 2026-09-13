// Kiosk stage control: Bambu-style XY jog pad, Z column with the ZEN-style
// focus visualization (tap/drag to move Z), live position readout, safe
// frame homing and an always-visible STOP.
import { useCallback, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  LinearProgress,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import ArrowUpwardRoundedIcon from "@mui/icons-material/ArrowUpwardRounded";
import ArrowDownwardRoundedIcon from "@mui/icons-material/ArrowDownwardRounded";
import ArrowBackRoundedIcon from "@mui/icons-material/ArrowBackRounded";
import ArrowForwardRoundedIcon from "@mui/icons-material/ArrowForwardRounded";
import HomeRoundedIcon from "@mui/icons-material/HomeRounded";
import StopRoundedIcon from "@mui/icons-material/StopRounded";
import MyLocationRoundedIcon from "@mui/icons-material/MyLocationRounded";
import StraightenRoundedIcon from "@mui/icons-material/StraightenRounded";
import { enqueueSnackbar } from "notistack";

import MobilePage from "../components/MobilePage";
import SectionLabel from "../components/SectionLabel";
import StatusTile from "../components/StatusTile";
import ZStackVisualization from "../../axon/experiment-designer/ZStackVisualization";
import * as positionSlice from "../../state/slices/PositionSlice";
import { getHomingState } from "../../state/slices/HomingSlice";
import apiPositionerControllerMovePositionerXYZ from "../../backendapi/apiPositionerControllerMovePositionerXYZ";
import apiPositionerControllerStopAllAxes from "../../backendapi/apiPositionerControllerStopAllAxes";
import apiPositionerControllerStartFrameHoming from "../../backendapi/apiPositionerControllerStartFrameHoming";
import apiPositionerControllerCancelFrameHoming from "../../backendapi/apiPositionerControllerCancelFrameHoming";

const XY_STEPS = [1, 10, 100, 1000];
const Z_STEPS = [1, 10, 100];
const Z_VIEW_HALF_RANGE = 100; // µm shown above/below the re-center point

const JOG_BTN_SX = {
  minWidth: 76,
  minHeight: 76,
  borderRadius: 3,
  "& svg": { fontSize: 34 },
};

const MobileStagePage = () => {
  const dispatch = useDispatch();
  const positionState = useSelector(positionSlice.getPositionState);
  const homingState = useSelector(getHomingState);

  const xyStep = positionState.stepSizes?.X ?? 100;
  const zStep = positionState.stepSizes?.Z ?? 10;

  const [zCenter, setZCenter] = useState(null);
  const [homingDialogOpen, setHomingDialogOpen] = useState(false);

  // Anchor the Z visualization window the first time we render (and via the
  // re-center button). The live focal plane then moves inside this window.
  useEffect(() => {
    if (zCenter === null && Number.isFinite(positionState.z)) {
      setZCenter(positionState.z);
    }
  }, [positionState.z, zCenter]);

  const jog = useCallback((deltas) => {
    apiPositionerControllerMovePositionerXYZ({
      ...deltas,
      isAbsolute: false,
    }).catch(() => enqueueSnackbar("Stage move failed", { variant: "error" }));
  }, []);

  const moveToZ = useCallback((zAbs) => {
    apiPositionerControllerMovePositionerXYZ({
      z: zAbs,
      isAbsolute: true,
    }).catch(() => enqueueSnackbar("Z move failed", { variant: "error" }));
  }, []);

  const handleStop = () => {
    apiPositionerControllerStopAllAxes()
      .then(() => enqueueSnackbar("All axes stopped", { variant: "info" }))
      .catch(() => enqueueSnackbar("Stop failed", { variant: "error" }));
  };

  const handleStartHoming = () => {
    setHomingDialogOpen(false);
    apiPositionerControllerStartFrameHoming()
      .then(() => enqueueSnackbar("Safe homing started", { variant: "info" }))
      .catch(() => enqueueSnackbar("Could not start homing", { variant: "error" }));
  };

  const handleCancelHoming = () => {
    apiPositionerControllerCancelFrameHoming().catch(() =>
      enqueueSnackbar("Could not cancel homing", { variant: "error" }),
    );
  };

  const setXyStep = (value) => {
    dispatch(positionSlice.setStepSize({ axis: "X", value }));
    dispatch(positionSlice.setStepSize({ axis: "Y", value }));
  };
  const setZStepSize = (value) =>
    dispatch(positionSlice.setStepSize({ axis: "Z", value }));

  const zFirst = (zCenter ?? 0) - Z_VIEW_HALF_RANGE;
  const zLast = (zCenter ?? 0) + Z_VIEW_HALF_RANGE;

  return (
    <MobilePage
      title="Stage"
      subtitle="Jog XY, focus Z — one tap moves one step"
      action={
        <Button
          color="error"
          variant="contained"
          startIcon={<StopRoundedIcon />}
          onClick={handleStop}
          sx={{ minHeight: 56 }}
        >
          Stop
        </Button>
      }
    >
      {homingState.active && (
        <Paper
          variant="outlined"
          sx={{ p: 2, mb: 2.5, borderRadius: 3, borderColor: "warning.main" }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
            <Typography variant="subtitle1" sx={{ flex: 1 }}>
              Homing — {homingState.phase}
              {homingState.message ? ` · ${homingState.message}` : ""}
            </Typography>
            <Button size="small" color="warning" onClick={handleCancelHoming}>
              Cancel
            </Button>
          </Box>
          <LinearProgress color="warning" sx={{ height: 8, borderRadius: 4 }} />
        </Paper>
      )}

      <Box sx={{ display: "flex", gap: 2.5, flexWrap: "wrap" }}>
        {/* XY pad */}
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3, flex: "0 1 auto" }}>
          <SectionLabel>XY stage</SectionLabel>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "repeat(3, auto)",
              gap: 1.5,
              justifyContent: "center",
              mb: 2,
            }}
          >
            <Box />
            <Button variant="contained" sx={JOG_BTN_SX} onClick={() => jog({ y: xyStep })}>
              <ArrowUpwardRoundedIcon />
            </Button>
            <Box />
            <Button variant="contained" sx={JOG_BTN_SX} onClick={() => jog({ x: -xyStep })}>
              <ArrowBackRoundedIcon />
            </Button>
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                color: "text.secondary",
              }}
            >
              <StraightenRoundedIcon fontSize="small" />
              <Typography variant="caption" sx={{ fontWeight: 700 }}>
                {xyStep >= 1000 ? `${xyStep / 1000} mm` : `${xyStep} µm`}
              </Typography>
            </Box>
            <Button variant="contained" sx={JOG_BTN_SX} onClick={() => jog({ x: xyStep })}>
              <ArrowForwardRoundedIcon />
            </Button>
            <Box />
            <Button variant="contained" sx={JOG_BTN_SX} onClick={() => jog({ y: -xyStep })}>
              <ArrowDownwardRoundedIcon />
            </Button>
            <Box />
          </Box>
          <ToggleButtonGroup
            exclusive
            fullWidth
            value={xyStep}
            onChange={(e, v) => v !== null && setXyStep(v)}
          >
            {XY_STEPS.map((s) => (
              <ToggleButton key={s} value={s}>
                {s >= 1000 ? `${s / 1000}mm` : `${s}µm`}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Paper>

        {/* Z focus */}
        <Paper variant="outlined" sx={{ p: 2.5, borderRadius: 3, flex: "0 1 auto" }}>
          <SectionLabel>Z focus</SectionLabel>
          <Box sx={{ display: "flex", gap: 2 }}>
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: 1.5,
                justifyContent: "center",
              }}
            >
              <Button variant="contained" sx={JOG_BTN_SX} onClick={() => jog({ z: zStep })}>
                <ArrowUpwardRoundedIcon />
              </Button>
              <ToggleButtonGroup
                exclusive
                orientation="vertical"
                value={zStep}
                onChange={(e, v) => v !== null && setZStepSize(v)}
              >
                {Z_STEPS.map((s) => (
                  <ToggleButton key={s} value={s}>
                    {s}µm
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
              <Button variant="contained" sx={JOG_BTN_SX} onClick={() => jog({ z: -zStep })}>
                <ArrowDownwardRoundedIcon />
              </Button>
            </Box>
            <Box sx={{ width: 210 }}>
              {/* Borrowed from the experiment designer: live focal-plane view;
                  tapping/dragging inside commands an absolute Z move. */}
              <ZStackVisualization
                firstAbs={zFirst}
                lastAbs={zLast}
                currentAbs={positionState.z}
                slicesAbs={[]}
                onSeek={moveToZ}
                height={300}
              />
              <Button
                size="small"
                fullWidth
                startIcon={<MyLocationRoundedIcon />}
                onClick={() => setZCenter(positionState.z)}
              >
                Re-center view
              </Button>
            </Box>
          </Box>
        </Paper>

        {/* Position + homing */}
        <Box
          sx={{
            flex: 1,
            minWidth: 240,
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          <Box sx={{ display: "flex", gap: 2 }}>
            <StatusTile label="X" value={`${positionState.x?.toFixed(1)} µm`} />
            <StatusTile label="Y" value={`${positionState.y?.toFixed(1)} µm`} />
          </Box>
          <Box sx={{ display: "flex", gap: 2 }}>
            <StatusTile label="Z" value={`${positionState.z?.toFixed(1)} µm`} />
            <StatusTile label="A" value={`${positionState.a?.toFixed(1)} µm`} />
          </Box>
          <Button
            size="large"
            variant="outlined"
            startIcon={<HomeRoundedIcon />}
            disabled={homingState.active}
            onClick={() => setHomingDialogOpen(true)}
          >
            Home all axes (safe)
          </Button>
          {homingState.active && (
            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              {Object.entries(homingState.axes || {}).map(([axis, st]) => (
                <Chip
                  key={axis}
                  size="small"
                  label={`${axis}: ${st}`}
                  color={st === "done" ? "success" : st === "homing" ? "warning" : "default"}
                  variant={st === "idle" ? "outlined" : "filled"}
                />
              ))}
            </Box>
          )}
        </Box>
      </Box>

      <Dialog open={homingDialogOpen} onClose={() => setHomingDialogOpen(false)}>
        <DialogTitle>Home all axes?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The collision-safe homing sequence homes Z first, lifts the stage,
            then homes X and Y. Make sure nothing blocks the stage travel.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHomingDialogOpen(false)} sx={{ flex: 1 }}>
            Cancel
          </Button>
          <Button variant="contained" onClick={handleStartHoming} sx={{ flex: 1 }}>
            Start homing
          </Button>
        </DialogActions>
      </Dialog>
    </MobilePage>
  );
};

export default MobileStagePage;
