import React, { useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Button,
  ButtonGroup,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import AddLocationIcon from "@mui/icons-material/AddLocation";
import MyLocationIcon from "@mui/icons-material/MyLocation";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import DeleteIcon from "@mui/icons-material/Delete";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import FileUploadIcon from "@mui/icons-material/FileUpload";

import WellSelectorCanvas, { Mode } from "../WellSelectorCanvas";
import * as wsUtils from "../WellSelectorUtils";
import InfoPopup from "../InfoPopup";

import * as wellSelectorSlice from "../../state/slices/WellSelectorSlice";
import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as positionSlice from "../../state/slices/PositionSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * PositionsDimension - Position/well selection interface
 * 
 * Contains:
 * - Well plate / XY canvas
 * - Selection modes: Single, Area, Pattern, Imported
 * - Navigation actions (move camera, add/remove, calibrate)
 * 
 * Explicitly excludes: Channels, Z, Time, Autofocus parameters
 */
const PositionsDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  
  // Refs
  const canvasRef = useRef();
  const infoPopupRef = useRef();
  const fileInputRef = useRef();
  
  // Redux state
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const positionState = useSelector(positionSlice.getPositionState);
  
  // Update summary when points change
  React.useEffect(() => {
    const pointCount = experimentState.pointList?.length || 0;
    const summary = pointCount === 0 
      ? "No positions defined"
      : pointCount === 1
        ? "1 position"
        : `${pointCount} positions`;
    
    dispatch(experimentUISlice.setDimensionSummary({ 
      dimension: DIMENSIONS.POSITIONS, 
      summary 
    }));
    dispatch(experimentUISlice.setDimensionConfigured({ 
      dimension: DIMENSIONS.POSITIONS, 
      configured: pointCount > 0 
    }));
  }, [experimentState.pointList, dispatch]);

  // Mode change handler
  const handleModeChange = (mode) => {
    dispatch(wellSelectorSlice.setMode(mode));
  };

  // Layout change handler
  const handleLayoutChange = (event) => {
    const layoutName = event.target.value;
    const offsetX = wellSelectorState.layoutOffsetX || 0;
    const offsetY = wellSelectorState.layoutOffsetY || 0;
    
    let wellLayout;
    
    switch (layoutName) {
      case "Default":
        wellLayout = wsUtils.wellLayoutDefault;
        break;
      case "Heidstar 4x Histosample":
        wellLayout = wsUtils.wellLayoutDevelopment;
        break;
      case "Wellplate 384":
        wellLayout = wsUtils.generateWellLayout384({ offsetX, offsetY });
        break;
      case "DEP Chip":
        wellLayout = wsUtils.generateWellLayoutDEPChip({ offsetX, offsetY });
        break;
      case "Ropod":
        wellLayout = wsUtils.ropodLayout;
        break;
      default:
        return;
    }
    
    // Apply offsets
    wellLayout = wsUtils.applyLayoutOffset(wellLayout, offsetX, offsetY);
    dispatch(experimentSlice.setWellLayout(wellLayout));
  };

  // Add current position
  const handleAddCurrentPosition = () => {
    const currentX = positionState.x;
    const currentY = positionState.y;
    
    dispatch(experimentSlice.createPoint({
      x: currentX,
      y: currentY,
      name: `Position ${experimentState.pointList.length + 1}`,
      shape: ""
    }));
    
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(`Added position: X=${currentX}, Y=${currentY}`);
    }
  };

  // Save positions to CSV
  const handleSavePositions = useCallback(() => {
    const points = experimentState.pointList || [];
    if (points.length === 0) {
      if (infoPopupRef.current) {
        infoPopupRef.current.showMessage("No positions to save");
      }
      return;
    }

    // Build CSV header and rows
    const header = "name,x,y,wellId,areaType";
    const rows = points.map((p) =>
      `${p.name || ""},${p.x},${p.y},${p.wellId || ""},${p.areaType || ""}`
    );
    const csv = [header, ...rows].join("\n");

    // Trigger download
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "positions.csv";
    link.click();
    URL.revokeObjectURL(url);

    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage(`Saved ${points.length} positions to CSV`);
    }
  }, [experimentState.pointList]);

  // Load positions from CSV
  const handleLoadPositions = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileSelected = useCallback((event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        const lines = text.split("\n").filter((l) => l.trim().length > 0);
        if (lines.length < 2) {
          infoPopupRef.current?.showMessage("CSV file is empty or has no data rows");
          return;
        }

        // Parse header to find column indices
        const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
        const nameIdx = header.indexOf("name");
        const xIdx = header.indexOf("x");
        const yIdx = header.indexOf("y");
        const wellIdx = header.indexOf("wellid");
        const areaIdx = header.indexOf("areatype");

        if (xIdx === -1 || yIdx === -1) {
          infoPopupRef.current?.showMessage("CSV must have 'x' and 'y' columns");
          return;
        }

        // Parse data rows
        const newPoints = [];
        for (let i = 1; i < lines.length; i++) {
          const cols = lines[i].split(",").map((c) => c.trim());
          const x = parseFloat(cols[xIdx]);
          const y = parseFloat(cols[yIdx]);
          if (isNaN(x) || isNaN(y)) continue;

          newPoints.push({
            name: nameIdx >= 0 ? cols[nameIdx] || `Position ${i}` : `Position ${i}`,
            x,
            y,
            shape: "",
            wellId: wellIdx >= 0 ? cols[wellIdx] || "" : "",
            areaType: areaIdx >= 0 ? cols[areaIdx] || "" : "",
          });
        }

        dispatch(experimentSlice.setPointList(newPoints));
        infoPopupRef.current?.showMessage(`Loaded ${newPoints.length} positions from CSV`);
      } catch (err) {
        console.error("Failed to parse CSV:", err);
        infoPopupRef.current?.showMessage("Failed to parse CSV file");
      }
    };
    reader.readAsText(file);
    // Reset the input so the same file can be re-selected
    event.target.value = "";
  }, [dispatch]);

  // Remove single position
  const handleRemovePosition = useCallback((index) => {
    dispatch(experimentSlice.removePoint(index));
  }, [dispatch]);

  // Clear all positions
  const handleClearAll = useCallback(() => {
    dispatch(experimentSlice.setPointList([]));
    if (infoPopupRef.current) {
      infoPopupRef.current.showMessage("All positions cleared");
    }
  }, [dispatch]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Info message about Well Selector tab */}
      <Box
        sx={{
          p: 2,
          mb: 2,
          border: `1px solid ${theme.palette.info.main}`,
          borderRadius: 1,
          backgroundColor: alpha(theme.palette.info.main, 0.08),
        }}
      >
        <Typography variant="body2" sx={{ mb: 0.5, fontWeight: 600 }}>
          📍 Position Selection
        </Typography>
        <Typography variant="body2" color="textSecondary">
          To select positions on the well plate, please switch to the <strong>Well Selector</strong> tab.
          The positions you define there will appear in this experiment.
        </Typography>
      </Box>

      {/* Quick add current position */}
      <Box sx={{ mb: 2 }}>
        <Button
          size="small"
          variant="outlined"
          startIcon={<AddLocationIcon />}
          onClick={handleAddCurrentPosition}
          fullWidth
        >
          Add Current Stage Position
        </Button>
      </Box>

      {/* Save / Load Positions as CSV */}
      <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
        <Button
          size="small"
          variant="outlined"
          fullWidth
          startIcon={<SaveAltIcon />}
          onClick={handleSavePositions}
          disabled={!experimentState.pointList || experimentState.pointList.length === 0}
        >
          Save Positions (CSV)
        </Button>
        <Button
          size="small"
          variant="outlined"
          fullWidth
          startIcon={<FileUploadIcon />}
          onClick={handleLoadPositions}
        >
          Load Positions (CSV)
        </Button>
        {/* Hidden file input for CSV loading */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.txt"
          style={{ display: "none" }}
          onChange={handleFileSelected}
        />
      </Box>

      {/* Position list table */}
      {experimentState.pointList && experimentState.pointList.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Positions ({experimentState.pointList.length})
            </Typography>
            <Button
              size="small"
              color="error"
              variant="text"
              onClick={handleClearAll}
            >
              Clear All
            </Button>
          </Box>
          <TableContainer sx={{ maxHeight: 250, border: `1px solid ${theme.palette.divider}`, borderRadius: 1 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }}>#</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }}>Name</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }} align="right">X (µm)</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }} align="right">Y (µm)</TableCell>
                  <TableCell sx={{ fontWeight: 600, py: 0.5 }} align="center">Well</TableCell>
                  <TableCell sx={{ py: 0.5 }} align="center"></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {experimentState.pointList.map((point, idx) => (
                  <TableRow key={point.id || idx} hover>
                    <TableCell sx={{ py: 0.25 }}>{idx + 1}</TableCell>
                    <TableCell sx={{ py: 0.25 }}>{point.name || "-"}</TableCell>
                    <TableCell sx={{ py: 0.25 }} align="right">{Math.round(point.x)}</TableCell>
                    <TableCell sx={{ py: 0.25 }} align="right">{Math.round(point.y)}</TableCell>
                    <TableCell sx={{ py: 0.25 }} align="center">
                      {point.wellId ? (
                        <Chip label={point.wellId} size="small" variant="outlined" sx={{ fontSize: "0.65rem", height: 18 }} />
                      ) : "-"}
                    </TableCell>
                    <TableCell sx={{ py: 0.25 }} align="center">
                      <IconButton size="small" onClick={() => handleRemovePosition(idx)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
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
