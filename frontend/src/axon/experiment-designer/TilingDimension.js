import React, { useEffect, useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import * as wellSelectorSlice from "../../state/slices/WellSelectorSlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * TilingDimension - Tiling/mosaic configuration interface
 *
 * Contains:
 * - Overlap X / Y
 * - Scan order (snake/raster)
 * - Stitching intent (preview / full / none)
 */
const TilingDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const wellSelectorState = useSelector(wellSelectorSlice.getWellSelectorState);
  const parameterValue = experimentState.parameterValue;

  // Calculate overlap percentage for display
  const overlapPercent = Math.round((parameterValue.overlapWidth || 0) * 100);

  // Update summary when parameters change
  useEffect(() => {
    const summary = `X/Y Overlap ${overlapPercent}%`;
    dispatch(experimentUISlice.setDimensionSummary({
      dimension: DIMENSIONS.TILING,
      summary,
    }));
  }, [overlapPercent, dispatch]);

  // Sync overlap values
  const handleOverlapChange = (value) => {
    const overlapValue = value / 100;
    // Update both experiment and well selector states
    dispatch(experimentSlice.setOverlapWidth(overlapValue));
    dispatch(experimentSlice.setOverlapHeight(overlapValue));
    dispatch(wellSelectorSlice.setOverlapWidth(overlapValue));
    dispatch(wellSelectorSlice.setOverlapHeight(overlapValue));
  };

  // Handle snakescan toggle
  const handleSnakescanChange = (checked) => {
    dispatch(experimentSlice.setIsSnakescan(checked));
    dispatch(wellSelectorSlice.setAreaSelectSnakescan(checked));
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Overlap Control */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          Tile Overlap
        </Typography>
        
        <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
          Set the overlap between adjacent tiles. Higher overlap improves stitching quality but increases acquisition time.
        </Typography>

        {/* Overlap slider */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <Slider
            value={overlapPercent}
            min={-1000}
            max={50}
            step={5}
            onChange={(e, val) => handleOverlapChange(val)}
            marks={[
              { value: 0, label: "0%" },
              { value: 10, label: "10%" },
              { value: 20, label: "20%" },
              { value: 30, label: "30%" },
              { value: 40, label: "40%" },
              { value: 50, label: "50%" },
            ]}
            sx={{ flex: 1, maxWidth: 400 }}
          />
          <Chip
            label={`${overlapPercent}%`}
            color="primary"
            variant="outlined"
            size="small"
          />
        </Box>

        {/* Quick presets */}
        <Box sx={{ display: "flex", gap: 0.5, mt: 1.5 }}>
          {[
            { label: "None", value: 0 },
            { label: "Minimal (10%)", value: 10 },
            { label: "Standard (20%)", value: 20 },
            { label: "High (30%)", value: 30 },
          ].map((preset) => (
            <Chip
              key={preset.value}
              label={preset.label}
              size="small"
              onClick={() => handleOverlapChange(preset.value)}
              color={overlapPercent === preset.value ? "primary" : "default"}
              variant={overlapPercent === preset.value ? "filled" : "outlined"}
              sx={{ cursor: "pointer" }}
            />
          ))}
        </Box>
      </Box>

      {/* Scan Pattern */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          Scan Pattern
        </Typography>

        <Box
          sx={{
            display: "flex",
            gap: 2,
            p: 2,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
          }}
        >
          {/* Raster pattern */}
          <Box
            onClick={() => handleSnakescanChange(false)}
            sx={{
              flex: 1,
              p: 2,
              textAlign: "center",
              borderRadius: 1,
              cursor: "pointer",
              border: `2px solid ${!parameterValue.is_snakescan ? theme.palette.primary.main : "transparent"}`,
              backgroundColor: !parameterValue.is_snakescan 
                ? alpha(theme.palette.primary.main, 0.08) 
                : "transparent",
              "&:hover": {
                backgroundColor: alpha(theme.palette.action.hover, 0.1),
              },
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Raster
            </Typography>
            <Box sx={{ fontFamily: "monospace", fontSize: "0.8rem", color: theme.palette.text.secondary }}>
              → → → →<br />
              → → → →<br />
              → → → →
            </Box>
            <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: "block" }}>
              Same direction each row
            </Typography>
          </Box>

          {/* Snake pattern */}
          <Box
            onClick={() => handleSnakescanChange(true)}
            sx={{
              flex: 1,
              p: 2,
              textAlign: "center",
              borderRadius: 1,
              cursor: "pointer",
              border: `2px solid ${parameterValue.is_snakescan ? theme.palette.primary.main : "transparent"}`,
              backgroundColor: parameterValue.is_snakescan 
                ? alpha(theme.palette.primary.main, 0.08) 
                : "transparent",
              "&:hover": {
                backgroundColor: alpha(theme.palette.action.hover, 0.1),
              },
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
              Snake
            </Typography>
            <Box sx={{ fontFamily: "monospace", fontSize: "0.8rem", color: theme.palette.text.secondary }}>
              → → → →<br />
              ← ← ← ←<br />
              → → → →
            </Box>
            <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: "block" }}>
              Alternating direction (faster)
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Stitching Intent */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
          Stitching
        </Typography>

        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel>Stitching Mode</InputLabel>
          <Select
            value={parameterValue.ome_write_stitched_tiff ? "full" : "none"}
            onChange={(e) => {
              dispatch(experimentSlice.setOmeWriteStitchedTiff(e.target.value === "full"));
            }}
            label="Stitching Mode"
          >
            <MenuItem value="none">
              <Box>
                <Typography variant="body2">None</Typography>
                <Typography variant="caption" color="textSecondary">
                  Save tiles individually
                </Typography>
              </Box>
            </MenuItem>
            <MenuItem value="full">
              <Box>
                <Typography variant="body2">Full Stitch</Typography>
                <Typography variant="caption" color="textSecondary">
                  Combine into single large image
                </Typography>
              </Box>
            </MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Advanced Settings */}
      <Accordion
        disableGutters
        sx={{
          boxShadow: "none",
          border: `1px solid ${theme.palette.divider}`,
          "&:before": { display: "none" },
        }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2">Advanced Settings</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {/* Independent X/Y overlap */}
            <Typography variant="caption" color="textSecondary">
              Independent X/Y overlap (advanced)
            </Typography>
            
            <Box sx={{ display: "flex", gap: 2 }}>
              <Box sx={{ flex: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Width Overlap
                </Typography>
                <Slider
                  value={Math.round(parameterValue.overlapWidth * 100)}
                  min={-1000}
                  max={50}
                  step={5}
                  onChange={(e, val) => {
                    dispatch(experimentSlice.setOverlapWidth(val / 100));
                    dispatch(wellSelectorSlice.setOverlapWidth(val / 100));
                  }}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(val) => `${val}%`}
                />
                <Typography variant="caption" color="textSecondary">
                  Negative = gap, Positive = overlap
                </Typography>
              </Box>

              <Box sx={{ flex: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  Height Overlap
                </Typography>
                <Slider
                  value={Math.round(parameterValue.overlapHeight * 100)}
                  min={-1000}
                  max={50}
                  step={5}
                  onChange={(e, val) => {
                    dispatch(experimentSlice.setOverlapHeight(val / 100));
                    dispatch(wellSelectorSlice.setOverlapHeight(val / 100));
                  }}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(val) => `${val}%`}
                />
                <Typography variant="caption" color="textSecondary">
                  Negative = gap, Positive = overlap
                </Typography>
              </Box>
            </Box>

            {/* Stage speed */}
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 500 }}>
                Stage Speed
              </Typography>
              <FormControl size="small" fullWidth sx={{ mt: 0.5 }}>
                <Select
                  value={parameterValue.speed || 20000}
                  onChange={(e) => dispatch(experimentSlice.setSpeed(Number(e.target.value)))}
                >
                  {[5000, 10000, 15000, 20000, 25000, 30000].map((speed) => (
                    <MenuItem key={speed} value={speed}>
                      {speed} µm/s
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default TilingDimension;
