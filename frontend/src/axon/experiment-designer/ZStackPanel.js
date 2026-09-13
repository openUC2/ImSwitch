import React, { useState, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Tabs,
  Tab,
  Button,
  IconButton,
  Chip,
  Stack,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
  Slider,
  Alert,
  Divider,
  Tooltip,
  useTheme,
  alpha,
} from "@mui/material";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import AnchorIcon from "@mui/icons-material/Anchor";
import AddIcon from "@mui/icons-material/Add";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import SwapVertIcon from "@mui/icons-material/SwapVert";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";
import apiPositionerControllerMovePositionerXYZ from "../../backendapi/apiPositionerControllerMovePositionerXYZ";
import FreeNumberField from "../../components/FreeNumberField";
import ZStackVisualization from "./ZStackVisualization";

const UI_TABS = { RANGE: "range", INDIVIDUAL: "individual", TABLE: "table" };

/**
 * ZStackPanel — ZEN-style Z-Stack configuration: an oblique animation of the
 * focal plane within the sample, First/Last bounds that mutually recompute
 * against slice count/interval, a position slider, and (Individual/Table
 * tabs) hand-picked non-uniform Z planes.
 *
 * The backend only ever receives RELATIVE offsets (zStackMin/zStackMax/
 * zStackOffsets) — see ExperimentController.py, which adds them on top of
 * each tile's own Z base (current Z, per-point Z, or focus-map Z). To show
 * ZEN-style absolute µm here without changing that contract, this panel
 * anchors the display to a live-captured Z (experimentUI.zStackAnchorZ),
 * set the first time the user presses "Set First"/"Set Last"/"Add current Z".
 */
const ZStackPanel = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  const experimentState = useSelector(experimentSlice.getExperimentState);
  const experimentUI = useSelector(experimentUISlice.getExperimentUIState);
  const positionState = useSelector(positionSlice.getPositionState);

  const parameterValue = experimentState.parameterValue;
  const anchorZ = experimentUI.dimensions[DIMENSIONS.Z_FOCUS]?.zStackAnchorZ;
  const anchored = anchorZ !== null && anchorZ !== undefined;

  const [uiTab, setUiTab] = useState(
    parameterValue.zStackMode === "individual" ? UI_TABS.INDIVIDUAL : UI_TABS.RANGE,
  );
  // Which of {Interval, Slices} stays fixed when First/Last change. UI-only
  // preference (like ZEN's "Keep" radio), not persisted with the experiment.
  const [keepMode, setKeepMode] = useState("interval");
  const [nudgeStep, setNudgeStep] = useState(5);

  const toAbs = useCallback((offset) => (anchored ? anchorZ + offset : offset), [anchored, anchorZ]);
  const toOffset = useCallback((abs) => (anchored ? abs - anchorZ : abs), [anchored, anchorZ]);

  // Anchors on first use (if not yet anchored) and returns the relative
  // offset for the live stage Z — shared by Set First / Set Last / Add slice.
  const captureCurrentAsOffset = useCallback(() => {
    const currentZ = positionState.z;
    if (!anchored) {
      dispatch(experimentUISlice.setZStackAnchorZ(currentZ));
      return 0;
    }
    return currentZ - anchorZ;
  }, [anchored, anchorZ, positionState.z, dispatch]);

  const handleReanchor = useCallback(() => {
    const newAnchor = positionState.z;
    if (anchored) {
      // Preserve every currently-shown absolute value under the new anchor.
      const firstAbsCurrent = anchorZ + parameterValue.zStackMin;
      const lastAbsCurrent = anchorZ + parameterValue.zStackMax;
      dispatch(experimentSlice.setZStackMin(firstAbsCurrent - newAnchor));
      dispatch(experimentSlice.setZStackMax(lastAbsCurrent - newAnchor));
      if (parameterValue.zStackOffsets.length) {
        dispatch(
          experimentSlice.setZStackOffsets(
            parameterValue.zStackOffsets.map((o) => anchorZ + o - newAnchor),
          ),
        );
      }
    }
    dispatch(experimentUISlice.setZStackAnchorZ(newAnchor));
  }, [anchored, anchorZ, parameterValue.zStackMin, parameterValue.zStackMax, parameterValue.zStackOffsets, positionState.z, dispatch]);

  const range = Math.abs(parameterValue.zStackMax - parameterValue.zStackMin);
  const stepSize = parameterValue.zStackStepSize || 1;
  const slices = Math.max(1, Math.round(range / stepSize) + 1);

  // After First/Last change, keep either Interval or Slices fixed per keepMode.
  const reconcileAfterBoundsChange = useCallback(
    (newMin, newMax) => {
      if (keepMode !== "slices") return; // "interval" kept as-is; Slices auto-derives.
      const newRange = Math.abs(newMax - newMin);
      const newStep = slices > 1 ? newRange / (slices - 1) : newRange;
      dispatch(experimentSlice.setZStackStepSize(newStep || 0.01));
    },
    [keepMode, slices, dispatch],
  );

  const handleSetFirst = () => {
    const offset = captureCurrentAsOffset();
    dispatch(experimentSlice.setZStackMin(offset));
    reconcileAfterBoundsChange(offset, parameterValue.zStackMax);
  };
  const handleSetLast = () => {
    const offset = captureCurrentAsOffset();
    dispatch(experimentSlice.setZStackMax(offset));
    reconcileAfterBoundsChange(parameterValue.zStackMin, offset);
  };
  const handleFirstAbsCommit = (newAbs) => {
    const offset = toOffset(newAbs);
    dispatch(experimentSlice.setZStackMin(offset));
    reconcileAfterBoundsChange(offset, parameterValue.zStackMax);
  };
  const handleLastAbsCommit = (newAbs) => {
    const offset = toOffset(newAbs);
    dispatch(experimentSlice.setZStackMax(offset));
    reconcileAfterBoundsChange(parameterValue.zStackMin, offset);
  };
  const handleSlicesCommit = (newSlicesRaw) => {
    const newSlices = Math.max(1, Math.round(newSlicesRaw));
    const newStep = newSlices > 1 ? range / (newSlices - 1) : range;
    dispatch(experimentSlice.setZStackStepSize(newStep || 0.01));
  };
  const handleIntervalCommit = (newInterval) => {
    if (newInterval > 0) dispatch(experimentSlice.setZStackStepSize(newInterval));
  };

  const handleAddCurrentZSlice = () => {
    dispatch(experimentSlice.addZStackOffset(captureCurrentAsOffset()));
  };

  const moveToZ = useCallback(
    (zAbs) => {
      apiPositionerControllerMovePositionerXYZ({
        z: zAbs,
        isAbsolute: true,
        speed: parameterValue.z_speed || 3000,
      }).catch((error) => console.error("[ZStackPanel] Error moving Z:", error));
    },
    [parameterValue.z_speed],
  );

  const applyNudge = (delta) => {
    if (!delta) return;
    dispatch(experimentSlice.setZStackMin(parameterValue.zStackMin + delta));
    dispatch(experimentSlice.setZStackMax(parameterValue.zStackMax + delta));
    if (parameterValue.zStackOffsets.length) {
      dispatch(
        experimentSlice.setZStackOffsets(parameterValue.zStackOffsets.map((o) => o + delta)),
      );
    }
  };

  const handleTabChange = (event, newTab) => {
    setUiTab(newTab);
    dispatch(
      experimentSlice.setZStackMode(newTab === UI_TABS.RANGE ? "range" : "individual"),
    );
  };

  const firstAbs = toAbs(parameterValue.zStackMin);
  const lastAbs = toAbs(parameterValue.zStackMax);
  const rangeSlicesAbs =
    parameterValue.zStackMode === "range"
      ? Array.from({ length: slices }, (_, i) =>
          toAbs(parameterValue.zStackMin + i * stepSize),
        )
      : [];
  const individualSlicesAbs = parameterValue.zStackOffsets.map(toAbs);
  const sortedIndividualWithIndex = parameterValue.zStackOffsets
    .map((v, i) => ({ v, i }))
    .sort((a, b) => a.v - b.v);

  const sliderMin = Math.min(firstAbs, lastAbs);
  const sliderMax = Math.max(firstAbs, lastAbs);

  return (
    <Box>
      {/* Anchor status + global Z offset */}
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 2,
          mb: 2,
          p: 1.5,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.background.default, 0.5),
        }}
      >
        <Chip
          size="small"
          color={anchored ? "primary" : "default"}
          variant={anchored ? "filled" : "outlined"}
          label={anchored ? `Anchored @ ${anchorZ.toFixed(2)} µm` : "Not yet anchored"}
        />
        <Tooltip title="Anchor First/Last/slices to the current stage Z, preserving today's absolute positions.">
          <span>
            <Button size="small" startIcon={<AnchorIcon />} onClick={handleReanchor}>
              Re-anchor to current Z
            </Button>
          </span>
        </Tooltip>

        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />

        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          <SwapVertIcon fontSize="small" sx={{ verticalAlign: "middle", mr: 0.5 }} />
          Global Z Offset
        </Typography>
        <IconButton size="small" onClick={() => applyNudge(-nudgeStep)} title={`Shift stack down ${nudgeStep} µm`}>
          <Typography variant="body2">−</Typography>
        </IconButton>
        <FreeNumberField
          label="Step (µm)"
          value={nudgeStep}
          onCommit={(v) => setNudgeStep(Math.abs(v) || 5)}
          sx={{ width: 100 }}
        />
        <IconButton size="small" onClick={() => applyNudge(nudgeStep)} title={`Shift stack up ${nudgeStep} µm`}>
          <AddIcon fontSize="small" />
        </IconButton>
        <Typography variant="caption" color="textSecondary">
          Shifts First/Last (and any individual slices) together — use if the sample has moved in Z.
        </Typography>
      </Box>

      {parameterValue.overrideZWithCurrentZ && (
        <Alert severity="info" icon={<WarningAmberIcon />} sx={{ mb: 2 }}>
          "Override per-group Z with current Z" is enabled (Tiling tab). This range is still
          applied as an offset — it will be centred on whatever Z each position is overridden to.
        </Alert>
      )}

      <Tabs value={uiTab} onChange={handleTabChange} sx={{ mb: 2 }}>
        <Tab label="Range" value={UI_TABS.RANGE} />
        <Tab label="Individual Slices" value={UI_TABS.INDIVIDUAL} />
        <Tab label="Table" value={UI_TABS.TABLE} />
      </Tabs>

      {uiTab === UI_TABS.RANGE && (
        <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
          <ZStackVisualization
            firstAbs={firstAbs}
            lastAbs={lastAbs}
            currentAbs={positionState.z}
            slicesAbs={rangeSlicesAbs}
            onSeek={moveToZ}
          />

          <Box sx={{ flex: 1, minWidth: 260, display: "flex", flexDirection: "column", gap: 2 }}>
            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
              <Box>
                <FreeNumberField
                  label="First (µm)"
                  value={firstAbs}
                  onCommit={handleFirstAbsCommit}
                  fixedDecimals={2}
                  sx={{ width: 140 }}
                />
                <Button size="small" startIcon={<MyLocationIcon />} onClick={handleSetFirst} sx={{ mt: 0.5 }}>
                  Set First
                </Button>
              </Box>
              <Box>
                <FreeNumberField
                  label="Last (µm)"
                  value={lastAbs}
                  onCommit={handleLastAbsCommit}
                  fixedDecimals={2}
                  sx={{ width: 140 }}
                />
                <Button size="small" startIcon={<MyLocationIcon />} onClick={handleSetLast} sx={{ mt: 0.5 }}>
                  Set Last
                </Button>
              </Box>
            </Box>

            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", alignItems: "flex-start" }}>
              <Typography variant="body2" sx={{ minWidth: 100, pt: 1 }}>
                Range: <strong>{range.toFixed(2)} µm</strong>
              </Typography>
              <FreeNumberField
                label="Slices"
                value={slices}
                onCommit={handleSlicesCommit}
                min={1}
                sx={{ width: 100 }}
              />
              <FreeNumberField
                label="Interval (µm)"
                value={stepSize}
                onCommit={handleIntervalCommit}
                fixedDecimals={2}
                min={0.01}
                sx={{ width: 130 }}
              />
            </Box>

            <FormControl>
              <FormLabel sx={{ typography: "caption" }}>
                Keep fixed when First/Last change
              </FormLabel>
              <RadioGroup row value={keepMode} onChange={(e) => setKeepMode(e.target.value)}>
                <FormControlLabel value="interval" control={<Radio size="small" />} label="Interval" />
                <FormControlLabel value="slices" control={<Radio size="small" />} label="Slices" />
              </RadioGroup>
            </FormControl>

            <Box>
              <Typography variant="caption" color="textSecondary">
                Position (µm)
              </Typography>
              <Slider
                value={Math.min(sliderMax, Math.max(sliderMin, positionState.z))}
                min={sliderMin}
                max={sliderMax}
                step={stepSize}
                valueLabelDisplay="auto"
                onChangeCommitted={(e, v) => moveToZ(v)}
              />
            </Box>
          </Box>
        </Box>
      )}

      {uiTab === UI_TABS.INDIVIDUAL && (
        <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
          <ZStackVisualization
            firstAbs={firstAbs}
            lastAbs={lastAbs}
            currentAbs={positionState.z}
            slicesAbs={individualSlicesAbs}
            onSeek={moveToZ}
          />

          <Box sx={{ flex: 1, minWidth: 260, display: "flex", flexDirection: "column", gap: 2 }}>
            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
              <Box>
                <FreeNumberField
                  label="First (µm)"
                  value={firstAbs}
                  onCommit={handleFirstAbsCommit}
                  fixedDecimals={2}
                  sx={{ width: 140 }}
                />
                <Button size="small" startIcon={<MyLocationIcon />} onClick={handleSetFirst} sx={{ mt: 0.5 }}>
                  Set First
                </Button>
              </Box>
              <Box>
                <FreeNumberField
                  label="Last (µm)"
                  value={lastAbs}
                  onCommit={handleLastAbsCommit}
                  fixedDecimals={2}
                  sx={{ width: 140 }}
                />
                <Button size="small" startIcon={<MyLocationIcon />} onClick={handleSetLast} sx={{ mt: 0.5 }}>
                  Set Last
                </Button>
              </Box>
            </Box>

            <Button
              variant="outlined"
              size="small"
              startIcon={<AddIcon />}
              onClick={handleAddCurrentZSlice}
              sx={{ alignSelf: "flex-start" }}
            >
              Add current Z as slice
            </Button>

            <Stack direction="row" flexWrap="wrap" gap={1}>
              {sortedIndividualWithIndex.length === 0 && (
                <Typography variant="caption" color="textSecondary">
                  No slices yet — move the stage and click "Add current Z as slice".
                </Typography>
              )}
              {sortedIndividualWithIndex.map(({ v, i }) => (
                <Chip
                  key={i}
                  label={`${toAbs(v).toFixed(2)} µm`}
                  onDelete={() => dispatch(experimentSlice.removeZStackOffsetAt(i))}
                />
              ))}
            </Stack>
          </Box>
        </Box>
      )}

      {uiTab === UI_TABS.TABLE && (
        <Box>
          <TableContainer component={Paper} variant="outlined" sx={{ maxWidth: 420, mb: 2 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Offset (µm)</TableCell>
                  <TableCell>Absolute (µm)</TableCell>
                  <TableCell align="right" />
                </TableRow>
              </TableHead>
              <TableBody>
                {parameterValue.zStackOffsets.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={3}>
                      <Typography variant="caption" color="textSecondary">
                        No slices yet — use "Add row" below.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
                {parameterValue.zStackOffsets.map((offset, i) => (
                  <TableRow key={i}>
                    <TableCell>
                      <FreeNumberField
                        value={offset}
                        onCommit={(v) =>
                          dispatch(experimentSlice.updateZStackOffsetAt({ index: i, value: v }))
                        }
                        fixedDecimals={2}
                        sx={{ width: 110 }}
                      />
                    </TableCell>
                    <TableCell>
                      <FreeNumberField
                        value={toAbs(offset)}
                        onCommit={(v) =>
                          dispatch(
                            experimentSlice.updateZStackOffsetAt({ index: i, value: toOffset(v) }),
                          )
                        }
                        fixedDecimals={2}
                        sx={{ width: 110 }}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <IconButton
                        size="small"
                        onClick={() => dispatch(experimentSlice.removeZStackOffsetAt(i))}
                      >
                        <DeleteOutlineIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <Button
            variant="outlined"
            size="small"
            startIcon={<AddIcon />}
            onClick={() => dispatch(experimentSlice.addZStackOffset((parameterValue.zStackMin + parameterValue.zStackMax) / 2))}
          >
            Add row
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default ZStackPanel;
