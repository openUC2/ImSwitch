import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Typography,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Tooltip,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
} from "@mui/material";
import { useTheme, alpha } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import InfoIcon from "@mui/icons-material/Info";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import SpeedIcon from "@mui/icons-material/Speed";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";

import * as experimentSlice from "../../state/slices/ExperimentSlice";
import * as experimentUISlice from "../../state/slices/ExperimentUISlice";
import { DIMENSIONS } from "../../state/slices/ExperimentUISlice";

/**
 * OutputDimension - Output/file format configuration interface
 *
 * Contains:
 * - File format selection (OME-TIFF, OME-Zarr, etc.)
 * - Stitching output options
 * - Compression/chunking (hidden by default)
 * - Metadata preview (optional)
 */
const OutputDimension = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // Redux state
  const experimentState = useSelector(experimentSlice.getExperimentState);
  const parameterValue = experimentState.parameterValue;

  // Determine active formats
  const activeFormats = [];
  if (parameterValue.ome_write_zarr) activeFormats.push("OME-Zarr");
  if (parameterValue.ome_write_tiff) activeFormats.push("OME-TIFF");
  if (parameterValue.ome_write_stitched_tiff) activeFormats.push("Stitched TIFF");
  if (parameterValue.ome_write_individual_tiffs) activeFormats.push("Individual TIFFs");

  // Update summary when formats change
  useEffect(() => {
    const summary = activeFormats.length > 0 
      ? `Save as ${activeFormats.join(", ")}`
      : "Default output";
    
    dispatch(experimentUISlice.setDimensionSummary({
      dimension: DIMENSIONS.OUTPUT,
      summary,
    }));
    dispatch(experimentUISlice.setDimensionConfigured({
      dimension: DIMENSIONS.OUTPUT,
      configured: activeFormats.length > 0,
    }));
  }, [parameterValue, dispatch]);

  // Format options with descriptions
  const formatOptions = [
    {
      key: "ome_write_zarr",
      label: "OME-Zarr",
      description: "Cloud-optimized chunked format. Best for large datasets, remote access, and parallel processing.",
      recommended: true,
      value: parameterValue.ome_write_zarr,
      onChange: (checked) => dispatch(experimentSlice.setOmeWriteZarr(checked)),
    },
    {
      key: "ome_write_tiff",
      label: "OME-TIFF",
      description: "Standard microscopy format with full metadata. Compatible with ImageJ, FIJI, and other analysis software.",
      recommended: false,
      value: parameterValue.ome_write_tiff,
      onChange: (checked) => dispatch(experimentSlice.setOmeWriteTiff(checked)),
    },
    {
      key: "ome_write_stitched_tiff",
      label: "Stitched OME-TIFF",
      description: "Single large stitched image combining all tiles. Warning: Can produce very large files.",
      recommended: false,
      value: parameterValue.ome_write_stitched_tiff,
      onChange: (checked) => dispatch(experimentSlice.setOmeWriteStitchedTiff(checked)),
    },
    {
      key: "ome_write_individual_tiffs",
      label: "Individual TIFFs",
      description: "Separate TIFF file per tile with position-based naming. Useful for distributed processing.",
      recommended: false,
      value: parameterValue.ome_write_individual_tiffs,
      onChange: (checked) => dispatch(experimentSlice.setOmeWriteIndividualTiffs(checked)),
    },
  ];

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Performance Mode Section */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600, display: "flex", alignItems: "center", gap: 1 }}>
          <SpeedIcon fontSize="small" />
          Acquisition Mode
        </Typography>

        {/* Performance Mode Toggle */}
        <Box
          sx={{
            display: "flex",
            alignItems: "flex-start",
            p: 2,
            border: `2px solid ${parameterValue.performanceMode ? theme.palette.warning.main : theme.palette.divider}`,
            borderRadius: 1,
            backgroundColor: parameterValue.performanceMode 
              ? alpha(theme.palette.warning.main, 0.08) 
              : "transparent",
            mb: 2,
          }}
        >
          <Switch
            checked={parameterValue.performanceMode || false}
            onChange={(e) => dispatch(experimentSlice.setPerformanceMode(e.target.checked))}
            color="warning"
            sx={{ mr: 1.5, mt: -0.5 }}
          />
          
          <Box sx={{ flex: 1 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                Performance Mode (Hardware Triggering)
              </Typography>
              {parameterValue.performanceMode && (
                <Chip
                  label="Active"
                  size="small"
                  color="warning"
                  sx={{ fontSize: "0.65rem", height: "18px" }}
                />
              )}
            </Box>
            <Typography variant="caption" color="textSecondary">
              Offloads timing-critical operations to the microcontroller for maximum speed. 
              Stage movement, illumination switching, and camera triggering are handled by hardware.
            </Typography>
          </Box>
        </Box>

        {/* Performance Mode Settings (shown when enabled) */}
        {parameterValue.performanceMode && (
          <Box
            sx={{
              p: 2,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              backgroundColor: alpha(theme.palette.background.paper, 0.5),
            }}
          >
            {/* Trigger Mode Selection */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" sx={{ fontWeight: 500, mb: 1, display: "block" }}>
                Trigger Mode
              </Typography>
              <FormControl size="small" fullWidth>
                <Select
                  value={parameterValue.performanceTriggerMode || "hardware"}
                  onChange={(e) => dispatch(experimentSlice.setPerformanceTriggerMode(e.target.value))}
                >
                  <MenuItem value="hardware">
                    <Box>
                      <Typography variant="body2">Hardware Trigger (External)</Typography>
                      <Typography variant="caption" color="textSecondary">
                        Camera triggered via TTL signal from microcontroller
                      </Typography>
                    </Box>
                  </MenuItem>
                  <MenuItem value="software">
                    <Box>
                      <Typography variant="body2">Software Trigger (Callback)</Typography>
                      <Typography variant="caption" color="textSecondary">
                        Camera triggered via software when receiving {"{"}"cam":1{"}"} signal
                      </Typography>
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>
            </Box>

            {/* Timing Parameters */}
            <Box sx={{ display: "flex", gap: 2 }}>
              <Box sx={{ flex: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
                  tPre (Settle Time, time it takes to activate the trigger)
                </Typography>
                <TextField
                  type="number"
                  size="small"
                  fullWidth
                  value={parameterValue.performanceTPreMs || 90}
                  onChange={(e) => dispatch(experimentSlice.setPerformanceTPreMs(Number(e.target.value)))}
                  InputProps={{
                    endAdornment: <Typography variant="caption" sx={{ ml: 1 }}>ms</Typography>,
                  }}
                  inputProps={{ min: 0, step: 10 }}
                />
                <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5, display: "block" }}>
                  Time before exposure (stage settling)
                </Typography>
              </Box>

              <Box sx={{ flex: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 500, mb: 0.5, display: "block" }}>
                  tPost (Exposure, time until the next action)
                </Typography>
                <TextField
                  type="number"
                  size="small"
                  fullWidth
                  value={parameterValue.performanceTPostMs || 50}
                  onChange={(e) => dispatch(experimentSlice.setPerformanceTPostMs(Number(e.target.value)))}
                  InputProps={{
                    endAdornment: <Typography variant="caption" sx={{ ml: 1 }}>ms</Typography>,
                  }}
                  inputProps={{ min: 0, step: 10 }}
                />
                <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5, display: "block" }}>
                  Exposure/acquisition time
                </Typography>
              </Box>
            </Box>

            {/* Warning for software trigger mode */}
            {parameterValue.performanceTriggerMode === "software" && (
              <Box
                sx={{
                  mt: 2,
                  p: 1.5,
                  borderRadius: 1,
                  backgroundColor: alpha(theme.palette.info.main, 0.08),
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 1,
                }}
              >
                <InfoIcon sx={{ fontSize: 16, color: theme.palette.info.main, mt: 0.25 }} />
                <Typography variant="caption" color="textSecondary">
                  Software trigger mode uses callbacks from the microcontroller to trigger the camera. 
                  This allows for more flexibility but may have slightly higher latency than hardware triggering.
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Format Selection */}
      <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
        Output Formats
      </Typography>

      <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
        Select one or more output formats. Multiple formats can be saved simultaneously.
      </Typography>

      {/* Format Cards */}
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
        {formatOptions.map((format) => (
          <Box
            key={format.key}
            sx={{
              display: "flex",
              alignItems: "flex-start",
              p: 2,
              border: `1px solid ${format.value ? theme.palette.primary.main : theme.palette.divider}`,
              borderRadius: 1,
              backgroundColor: format.value 
                ? alpha(theme.palette.primary.main, 0.04) 
                : "transparent",
              transition: "all 0.2s ease-in-out",
            }}
          >
            <Switch
              checked={format.value || false}
              onChange={(e) => format.onChange(e.target.checked)}
              sx={{ mr: 1.5, mt: -0.5 }}
            />
            
            <Box sx={{ flex: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {format.label}
                </Typography>
                {format.recommended && (
                  <Chip
                    label="Recommended"
                    size="small"
                    color="success"
                    variant="outlined"
                    sx={{ fontSize: "0.65rem", height: "18px" }}
                  />
                )}
                {format.value && (
                  <CheckCircleIcon 
                    sx={{ fontSize: 16, color: theme.palette.success.main, ml: "auto" }} 
                  />
                )}
              </Box>
              <Typography variant="caption" color="textSecondary">
                {format.description}
              </Typography>
            </Box>
          </Box>
        ))}
      </Box>

      {/* Quick info about selected formats */}
      {activeFormats.length > 0 && (
        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: 1,
            backgroundColor: alpha(theme.palette.info.main, 0.08),
          }}
        >
          <Typography variant="caption" color="textSecondary">
            <strong>Selected:</strong> {activeFormats.join(" + ")}
          </Typography>
        </Box>
      )}

      {/* Warning if no format selected */}
      {activeFormats.length === 0 && (
        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: 1,
            backgroundColor: alpha(theme.palette.warning.main, 0.08),
          }}
        >
          <Typography variant="caption" color="warning.main">
            ⚠️ No output format selected. Images will only be displayed but not saved.
          </Typography>
        </Box>
      )}

      {/* Advanced Settings */}
      <Accordion
        disableGutters
        sx={{
          mt: 2,
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
            {/* Compression options */}
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 500, mb: 1, display: "block" }}>
                Compression (Zarr)
              </Typography>
              <Typography variant="caption" color="textSecondary">
                OME-Zarr files use blosc compression by default for optimal balance of speed and size.
              </Typography>
            </Box>

            {/* Chunking options */}
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 500, mb: 1, display: "block" }}>
                Chunking Strategy
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Data is automatically chunked for efficient access patterns. Default chunk size: 256x256 pixels.
              </Typography>
            </Box>

            {/* Metadata */}
            <Box>
              <Typography variant="caption" sx={{ fontWeight: 500, mb: 1, display: "block" }}>
                Metadata
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Full OME metadata is included automatically, containing acquisition parameters, 
                channel information, physical pixel sizes, and timestamps.
              </Typography>
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Metadata Preview (future feature placeholder) */}
      <Accordion
        disableGutters
        disabled
        sx={{
          mt: 1,
          boxShadow: "none",
          border: `1px solid ${theme.palette.divider}`,
          "&:before": { display: "none" },
          opacity: 0.5,
        }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="body2">Metadata Preview (Coming Soon)</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="caption" color="textSecondary">
            Preview the metadata that will be saved with your experiment.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default OutputDimension;
