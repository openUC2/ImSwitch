import React, { useRef, useCallback, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import AddLocationIcon from "@mui/icons-material/AddLocation";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import HeightIcon from "@mui/icons-material/Height";
import DeleteIcon from "@mui/icons-material/Delete";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import FileUploadIcon from "@mui/icons-material/FileUpload";

import InfoPopup from "../InfoPopup";
import apiPositionerControllerMovePositioner from "../../backendapi/apiPositionerControllerMovePositioner.js";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * PositionsDimension - Position list for the experiment.
 *
 * Positions are picked on the Plate Map viewport (always visible to the left in
 * the WellPlate workspace) and edited here: name + X/Y/Z are editable inline,
 * the stage can be driven to a row, and the list saves/loads as CSV. This is the
 * single position editor — the old standalone "Points" tab was merged into it.
 */
const PositionsDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  const infoPopupRef = useRef();
  const fileInputRef = useRef();

  const experimentState = useSelector(experimentSlice.getExperimentState);
  const positionState = useSelector(positionSlice.getPositionState);

  const pointList = useMemo(() => experimentState.pointList || [], [experimentState.pointList]);

  // Keep the dimension summary / configured flag in sync with the point count.
  React.useEffect(() => {
    const pointCount = pointList.length;
    const summary =
      pointCount === 0
        ? "No positions defined"
        : pointCount === 1
          ? "1 position"
          : `${pointCount} positions`;

    dispatch(experimentUISlice.setDimensionSummary({ dimension: DIMENSIONS.POSITIONS, summary }));
    dispatch(
      experimentUISlice.setDimensionConfigured({
        dimension: DIMENSIONS.POSITIONS,
        configured: pointCount > 0,
      }),
    );
  }, [pointList, dispatch]);

  // Add the current stage XY as a new point.
  const handleAddCurrentPosition = useCallback(() => {
    dispatch(
      experimentSlice.createPoint({
        x: positionState.x,
        y: positionState.y,
        z: positionState.z,
        name: `Position ${pointList.length + 1}`,
        shape: "",
      }),
    );
    infoPopupRef.current?.showMessage(`Added position: X=${positionState.x}, Y=${positionState.y}`);
  }, [dispatch, positionState.x, positionState.y, positionState.z, pointList.length]);

  // Inline-edit a single field of a point (name / x / y / z).
  const handlePointChanged = useCallback(
    (index, field, value) => {
      const updated = pointList.map((p, idx) => (idx === index ? { ...p, [field]: value } : p));
      dispatch(experimentSlice.setPointList(updated));
    },
    [pointList, dispatch],
  );

  // Number-field edit: ignore intermediate empty/NaN so pointList stays numeric.
  const handleNumberChanged = useCallback(
    (index, field, raw) => {
      if (raw === "") return;
      const num = parseFloat(raw);
      if (Number.isNaN(num)) return;
      handlePointChanged(index, field, num);
    },
    [handlePointChanged],
  );

  // Drive the stage to a stored position (absolute move). `includeZ` controls
  // whether Z is moved too: the plain move stays in-plane (XY only) so focus
  // isn't disturbed, while "incl. Z" also drives Z to the stored value.
  const handleGoto = useCallback((point, includeZ = false) => {
    apiPositionerControllerMovePositioner({ axis: "X", dist: point.x, isAbsolute: true, speed: 20000 }).catch(
      (e) => console.error("Goto X failed", e),
    );
    apiPositionerControllerMovePositioner({ axis: "Y", dist: point.y, isAbsolute: true, speed: 20000 }).catch(
      (e) => console.error("Goto Y failed", e),
    );
    if (includeZ && point.z != null && point.z !== "") {
      apiPositionerControllerMovePositioner({ axis: "Z", dist: point.z, isAbsolute: true, speed: 20000 }).catch(
        (e) => console.error("Goto Z failed", e),
      );
    }
    infoPopupRef.current?.showMessage(
      `Moving stage to ${point.name || "position"}${includeZ ? " (incl. Z)" : " (XY only)"}`,
    );
  }, []);

  const handleRemovePosition = useCallback(
    (index) => dispatch(experimentSlice.removePoint(index)),
    [dispatch],
  );

  const handleClearAll = useCallback(() => {
    dispatch(experimentSlice.setPointList([]));
    infoPopupRef.current?.showMessage("All positions cleared");
  }, [dispatch]);

  // Save positions to CSV (name + xyz + well/area so they round-trip).
  const handleSavePositions = useCallback(() => {
    if (pointList.length === 0) {
      infoPopupRef.current?.showMessage("No positions to save");
      return;
    }
    const header = "name,x,y,z,wellId,areaType";
    const rows = pointList.map(
      (p) => `${p.name || ""},${p.x},${p.y},${p.z ?? ""},${p.wellId || ""},${p.areaType || ""}`,
    );
    const csv = [header, ...rows].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "positions.csv";
    link.click();
    URL.revokeObjectURL(url);
    infoPopupRef.current?.showMessage(`Saved ${pointList.length} positions to CSV`);
  }, [pointList]);

  const handleLoadPositions = useCallback(() => fileInputRef.current?.click(), []);

  const handleFileSelected = useCallback(
    (event) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const lines = e.target.result.split("\n").filter((l) => l.trim().length > 0);
          if (lines.length < 2) {
            infoPopupRef.current?.showMessage("CSV file is empty or has no data rows");
            return;
          }
          const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
          const nameIdx = header.indexOf("name");
          const xIdx = header.indexOf("x");
          const yIdx = header.indexOf("y");
          const zIdx = header.indexOf("z");
          const wellIdx = header.indexOf("wellid");
          const areaIdx = header.indexOf("areatype");

          if (xIdx === -1 || yIdx === -1) {
            infoPopupRef.current?.showMessage("CSV must have 'x' and 'y' columns");
            return;
          }

          const newPoints = [];
          for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(",").map((c) => c.trim());
            const x = parseFloat(cols[xIdx]);
            const y = parseFloat(cols[yIdx]);
            if (Number.isNaN(x) || Number.isNaN(y)) continue;
            const z = zIdx >= 0 ? parseFloat(cols[zIdx]) : NaN;
            newPoints.push({
              name: nameIdx >= 0 ? cols[nameIdx] || `Position ${i}` : `Position ${i}`,
              x,
              y,
              ...(Number.isNaN(z) ? {} : { z }),
              shape: "",
              wellId: wellIdx >= 0 ? cols[wellIdx] || "" : "",
              areaType: areaIdx >= 0 ? cols[areaIdx] || "" : "",
            });
          }
          // Append imported positions to the existing list (don't replace).
          dispatch(experimentSlice.setPointList([...pointList, ...newPoints]));
          infoPopupRef.current?.showMessage(
            `Imported ${newPoints.length} position(s) — appended to ${pointList.length} existing.`,
          );
        } catch (err) {
          console.error("Failed to parse CSV:", err);
          infoPopupRef.current?.showMessage("Failed to parse CSV file");
        }
      };
      reader.readAsText(file);
      event.target.value = "";
    },
    [dispatch, pointList],
  );

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Hint: positions are selected on the always-visible Plate Map viewport */}
      <Box
        sx={{
          p: 1.5,
          mb: 2,
          border: `1px solid ${theme.palette.info.main}`,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.info.main, 0.08),
        }}
      >
        <Typography variant="body2" color="textSecondary">
          📍 Select positions on the <strong>Plate Map</strong> viewport (left). They appear here
          instantly — edit the name and X/Y/Z below, or drive the stage to a row.
        </Typography>
      </Box>

      {/* Add current stage position */}
      <Button
        size="small"
        variant="outlined"
        startIcon={<AddLocationIcon />}
        onClick={handleAddCurrentPosition}
        fullWidth
        sx={{ mb: 2 }}
      >
        Add Current Stage Position
      </Button>

      {/* Save / Load CSV */}
      <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
        <Tooltip title="Export all positions to a CSV file on disc">
          <span style={{ flex: 1, display: "flex" }}>
            <Button
              size="small"
              variant="outlined"
              fullWidth
              startIcon={<SaveAltIcon />}
              onClick={handleSavePositions}
              disabled={pointList.length === 0}
            >
              Export (CSV)
            </Button>
          </span>
        </Tooltip>
        <Tooltip title="Import positions from a CSV file and append them to the current list">
          <Button
            size="small"
            variant="outlined"
            fullWidth
            startIcon={<FileUploadIcon />}
            onClick={handleLoadPositions}
            sx={{ flex: 1 }}
          >
            Import (append)
          </Button>
        </Tooltip>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.txt"
          style={{ display: "none" }}
          onChange={handleFileSelected}
        />
      </Box>

      {/* Editable position list */}
      {pointList.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Positions ({pointList.length})
            </Typography>
            <Button size="small" color="error" variant="text" onClick={handleClearAll}>
              Clear All
            </Button>
          </Box>
          <TableContainer
            sx={{ maxHeight: 320, border: `1px solid ${theme.palette.divider}`, borderRadius: 1 }}
          >
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }}>#</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }}>Name</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }} align="right">X (µm)</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }} align="right">Y (µm)</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }} align="right">Z (µm)</TableCell>
                  <TableCell sx={{ py: 0.5 }} align="center"></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {pointList.map((point, idx) => (
                  <TableRow key={point.id || idx} hover>
                    <TableCell sx={{ py: 0.25 }}>{idx + 1}</TableCell>
                    <TableCell sx={{ py: 0.25, minWidth: 110 }}>
                      <TextField
                        value={point.name || ""}
                        onChange={(e) => handlePointChanged(idx, "name", e.target.value)}
                        variant="standard"
                        size="small"
                        fullWidth
                        InputProps={{ disableUnderline: true, sx: { fontSize: "0.8rem" } }}
                      />
                    </TableCell>
                    <TableCell sx={{ py: 0.25, width: 84 }} align="right">
                      <TextField
                        value={point.x ?? ""}
                        onChange={(e) => handleNumberChanged(idx, "x", e.target.value)}
                        type="number"
                        variant="standard"
                        size="small"
                        InputProps={{ disableUnderline: true, sx: { fontSize: "0.8rem" } }}
                        inputProps={{ style: { textAlign: "right" } }}
                      />
                    </TableCell>
                    <TableCell sx={{ py: 0.25, width: 84 }} align="right">
                      <TextField
                        value={point.y ?? ""}
                        onChange={(e) => handleNumberChanged(idx, "y", e.target.value)}
                        type="number"
                        variant="standard"
                        size="small"
                        InputProps={{ disableUnderline: true, sx: { fontSize: "0.8rem" } }}
                        inputProps={{ style: { textAlign: "right" } }}
                      />
                    </TableCell>
                    <TableCell sx={{ py: 0.25, width: 84 }} align="right">
                      <TextField
                        value={point.z ?? ""}
                        onChange={(e) => handleNumberChanged(idx, "z", e.target.value)}
                        type="number"
                        variant="standard"
                        size="small"
                        placeholder="—"
                        InputProps={{ disableUnderline: true, sx: { fontSize: "0.8rem" } }}
                        inputProps={{ style: { textAlign: "right" } }}
                      />
                    </TableCell>
                    <TableCell sx={{ py: 0.25, whiteSpace: "nowrap" }} align="center">
                      <Tooltip title="Move stage here — XY only (keeps current focus)">
                        <IconButton size="small" onClick={() => handleGoto(point, false)}>
                          <MyLocationIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Move stage here — including Z">
                        <IconButton size="small" onClick={() => handleGoto(point, true)}>
                          <HeightIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Remove position">
                        <IconButton size="small" onClick={() => handleRemovePosition(idx)}>
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      <InfoPopup ref={infoPopupRef} />
    </Box>
  );
};

export default PositionsDimension;
