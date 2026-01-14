import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../backendapi/createAxiosInstance';
import { FormControlLabel, Switch, Tooltip, IconButton, Button } from "@mui/material";
import { Info as InfoIcon } from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";

import * as experimentSlice from "../state/slices/ExperimentSlice.js";
import * as parameterRangeSlice from "../state/slices/ParameterRangeSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import fetchExperimentControllerGetCurrentExperimentParams from "../middleware/fetchExperimentControllerGetCurrentExperimentParams.js";
import fetchLaserControllerCurrentValues from "../middleware/fetchLaserControllerCurrentValues.js";
import apiFocusLockControllerGetCurrentFocusValue from "../backendapi/apiFocusLockControllerGetCurrentFocusValue.js";

const ParameterEditorComponent = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  // read parameter & range from Redux
  const parameterValue = useSelector((state) => state.experimentState.parameterValue);
  const parameterRange = useSelector(parameterRangeSlice.getParameterRangeState);
  const connectionSettingsState = useSelector(connectionSettingsSlice.getConnectionSettingsState);

  // fallback arrays
  const intensities = parameterValue.illuIntensities || [];
  const gains       = parameterValue.gains || [];
  const exposures   = parameterValue.exposureTimes || [];

  // Fill missing intensity/gain/exposure with zero if needed
  const initializedIntensities = parameterRange.illuSources.map(
    (_, idx) => intensities[idx] ?? 0
  );
  const initializedGains = parameterRange.illuSources.map(
    (_, idx) => gains[idx] ?? 0
  );
  const initializedExposures = parameterRange.illuSources.map(
    (_, idx) => exposures[idx] ?? 0
  );

  // performance mode fallback
  const initializedPerformanceMode = parameterValue.performanceMode || false;

  // fetch experiment params on mount
  useEffect(() => {
    fetchExperimentControllerGetCurrentExperimentParams(dispatch);
  }, [dispatch]);

  // fetch current laser intensity values from backend when laser sources are available
  useEffect(() => {
    if (parameterRange.illuSources.length > 0 && connectionSettingsState.ip && connectionSettingsState.apiPort) {
      fetchLaserControllerCurrentValues(dispatch, connectionSettingsState, parameterRange.illuSources);
    }
  }, [dispatch, parameterRange.illuSources, connectionSettingsState.ip, connectionSettingsState.apiPort]);

  // ensure intensities array size matches available sources
  useEffect(() => { // TODO: This should be done in the fetch middleware I guess? @marco
    const initIlluIntensityArray = parameterRange.illuSources.map(
      (_, idx) => intensities[idx] ?? 0
    );
    const initGainsArray = parameterRange.illuSources.map(
      (_, idx) => gains[idx] ?? 0
    );
    const initExposuresArray = parameterRange.illuSources.map(
      (_, idx) => exposures[idx] ?? 0
    );
    if (JSON.stringify(intensities) !== JSON.stringify(initIlluIntensityArray)) {
      dispatch(experimentSlice.setIlluminationIntensities(initIlluIntensityArray));
    }
    if (JSON.stringify(gains) !== JSON.stringify(initGainsArray)) {
      dispatch(experimentSlice.setGains(initGainsArray));
    }
    if (JSON.stringify(exposures) !== JSON.stringify(initExposuresArray)) {
      dispatch(experimentSlice.setExposureTimes(initExposuresArray));
    }
    // apply the illusources to the experiment state
    if (parameterRange.illuSources.length > 0) {
      dispatch(experimentSlice.setIllumination(parameterRange.illuSources));
    }
  }, [parameterRange.illuSources, intensities, dispatch]);

  // helper functions for intensities, gains, exposures
  const setIntensity = async (idx, value) => {
    const arr = [...intensities];
    arr[idx] = value;
    dispatch(experimentSlice.setIlluminationIntensities(arr));

    // Also update backend immediately for real-time feedback
    const laserName = parameterRange.illuSources[idx];
    if (laserName && connectionSettingsState.ip && connectionSettingsState.apiPort) {
      try {
        const api = createAxiosInstance();
        const encodedLaserName = encodeURIComponent(laserName);
        await api.get(
          `/LaserController/setLaserValue?laserName=${encodedLaserName}&value=${value}`
        );
      } catch (error) {
        console.error("Failed to update laser intensity in backend:", error);
        // Continue without blocking UI - backend update is optional for better UX
      }
    }
  };

  const setGains = (idx, value) => {
    const arr = [...gains];
    arr[idx] = value;
    dispatch(experimentSlice.setGains(arr));
  };

  const setExposure = (idx, value) => {
    const arr = [...exposures];
    arr[idx] = value;
    dispatch(experimentSlice.setExposureTimes(arr));
  };

  const setPerformanceMode = (mode) => {
    dispatch(experimentSlice.setPerformanceMode(mode));
  };

  // Function to poll current focus value from FocusLock controller
  const pollCurrentFocusValue = async () => {
    try {
      const result = await apiFocusLockControllerGetCurrentFocusValue();
      const focusValue = result.focus_value;
      dispatch(experimentSlice.setAutoFocusTargetSetpoint(focusValue));
      console.log(`Polled focus value: ${focusValue}, updated target setpoint`);
    } catch (error) {
      console.error("Failed to poll current focus value:", error);
      alert("Failed to get current focus value. Make sure FocusLock measurement is running.");
    }
  };

  const tdStyle = { padding: "6px 8px", verticalAlign: "top" };

  return (
    <div style={{ textAlign: "left", padding: 10, fontSize: 16 }}>
      <h4 style={{ margin: 0, fontSize: 30 }}>Parameter Editor</h4>
      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 10 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #ddd" }}>
            <th style={tdStyle}>Category</th>
            <th style={tdStyle}>Parameter</th>
            <th style={tdStyle}>Value</th>
          </tr>
        </thead>
        <tbody>
          {/* Displays all illumination sources without checkboxes */}
          {parameterRange.illuSources.map((src, idx) => {
            const minI = parameterRange.illuSourceMinIntensities[idx] || 0;
            const maxI = parameterRange.illuSourceMaxIntensities[idx] || 1023;
            return (
              <tr key={`illu-${src}`}>
                <td style={tdStyle}>Illumination</td>
                <td style={tdStyle}>{src}</td>
                <td style={tdStyle}>
                  {/* intensity slider */}
                  <div style={{ marginBottom: 4 }}>
                    <span style={{ marginRight: 6 }}>Intensity:</span>
                    <input
                      type="range"
                      min={minI}
                      max={maxI}
                      step="1"
                      value={initializedIntensities[idx]}
                      onChange={(e) => setIntensity(idx, Number(e.target.value))}
                      style={{ width: "60%" }}
                    />
                    <span style={{ marginLeft: 6 }}>
                      {initializedIntensities[idx]} mW
                    </span>
                  </div>

                  {/* gain */}
                  <div style={{ marginBottom: 4 }}>
                    <span style={{ marginRight: 6 }}>Gain:</span>
                    <input
                      type="number"
                      value={initializedGains[idx]}
                      onChange={(e) => setGains(idx, Number(e.target.value))}
                      style={{ width: 80 }}
                      min={0}
                      max={23}
                    />
                  </div>

                  {/* exposure */}
                  <div style={{ marginBottom: 4 }}>
                    <span style={{ marginRight: 6 }}>Exposure (ms):</span>
                    <input
                      type="number"
                      value={initializedExposures[idx]}
                      onChange={(e) => setExposure(idx, Number(e.target.value))}
                      style={{ width: 80 }}
                    />
                  </div>

                  {/* performance mode toggle */}
                  <div>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={initializedPerformanceMode}
                          onChange={(e) => setPerformanceMode(e.target.checked)}
                        />
                      }
                      label="Performance Mode"
                    />
                  </div>
                </td>
              </tr>
            );
          })}

          {/* Speed */}
          <tr>
            <td></td>
            <td style={tdStyle}>Speed</td>
            <td style={tdStyle}>
              <select
                value={parameterValue.speed}
                onChange={(e) =>
                  dispatch(experimentSlice.setSpeed(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              >
                {parameterRange.speed.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </td>
          </tr>

          {/* Time-lapse */}
          <tr>
            <td rowSpan="2" style={tdStyle}>
              Time-lapse
            </td>
            <td style={tdStyle}>Period (s)</td>
            <td style={tdStyle}>
              <input
                type="number"
                min={parameterRange.timeLapsePeriod.min}
                max={parameterRange.timeLapsePeriod.max}
                step="1"
                value={parameterValue.timeLapsePeriod}
                onChange={(e) =>
                  dispatch(
                    experimentSlice.setTimeLapsePeriod(Number(e.target.value))
                  )
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Number of Images</td>
            <td style={tdStyle}>
              <input
                type="number"
                min={parameterRange.numberOfImages.min}
                max={parameterRange.numberOfImages.max}
                step="1"
                value={parameterValue.numberOfImages}
                onChange={(e) =>
                  dispatch(experimentSlice.setNumberOfImages(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>

          {/* Autofocus */}
          <tr>
            <td rowSpan="16" style={tdStyle}>
              Autofocus
            </td>
            <td style={tdStyle}>Enable</td>
            <td style={tdStyle}>
              <input
                type="checkbox"
                checked={parameterValue.autoFocus}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocus(e.target.checked))
                }
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>
              Autofocus Mode
              <Tooltip title="Software: Z-sweep autofocus (scans through Z positions). Hardware: One-shot autofocus using FocusLock controller (requires calibrated FocusLock).">
                <IconButton size="small">
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </td>
            <td style={tdStyle}>
              <select
                value={parameterValue.autoFocusMode || "software"}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusMode(e.target.value))
                }
                style={{ width: "100%", padding: 5 }}
              >
                <option value="software">Software (Z-Sweep)</option>
                <option value="hardware">Hardware (FocusLock One-Shot)</option>
              </select>
            </td>
          </tr>
          {/* Hardware autofocus specific parameters */}
          {parameterValue.autoFocusMode === "hardware" && (
            <>
              <tr>
                <td style={tdStyle}>Max Attempts</td>
                <td style={tdStyle}>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    max="10"
                    value={parameterValue.autofocus_max_attempts || 3}
                    onChange={(e) =>
                      dispatch(experimentSlice.setAutoFocusMaxAttempts(Number(e.target.value)))
                    }
                    style={{ width: "100%", padding: 5 }}
                  />
                </td>
              </tr>
              <tr>
                <td style={tdStyle}>
                  Target Focus Setpoint
                  <Tooltip title="Enter the target focus value manually or poll the current value from FocusLock controller">
                    <IconButton size="small">
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </td>
                <td style={tdStyle}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <input
                      type="number"
                      step="0.1"
                      value={parameterValue.autofocus_target_focus_setpoint || 0}
                      onChange={(e) =>
                        dispatch(experimentSlice.setAutoFocusTargetSetpoint(Number(e.target.value)))
                      }
                      style={{ flex: 1, padding: 5 }}
                    />
                    <Button 
                      variant="outlined" 
                      size="small" 
                      onClick={pollCurrentFocusValue}
                      style={{ whiteSpace: 'nowrap' }}
                    >
                      📊 Poll Current
                    </Button>
                  </div>
                </td>
              </tr>
            </>
          )}
          <tr>
            <td style={tdStyle}>Illumination Channel</td>
            <td style={tdStyle}>
              <select
                value={parameterValue.autoFocusIlluminationChannel || ""}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusIlluminationChannel(e.target.value))
                }
                style={{ width: "100%", padding: 5 }}
              >
                <option value="">Auto (use active channel)</option>
                {parameterRange.illuSources.map((source) => (
                  <option key={source} value={source}>
                    {source}
                  </option>
                ))}
              </select>
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Settle Time (s)</td>
            <td style={tdStyle}>
              <input
                type="number"
                step="0.01"
                min="0"
                max="10"
                value={parameterValue.autoFocusSettleTime || 0.1}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusSettleTime(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Range (±µm)</td>
            <td style={tdStyle}>
              <input
                type="number"
                step="1"
                min="1"
                value={parameterValue.autoFocusRange || 100}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusRange(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Resolution (µm)</td>
            <td style={tdStyle}>
              <input
                type="number"
                step="0.1"
                min="0.1"
                value={parameterValue.autoFocusResolution || 10}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusResolution(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Crop Size (px)</td>
            <td style={tdStyle}>
              <input
                type="number"
                step="128"
                min="256"
                max="4096"
                value={parameterValue.autoFocusCropsize || 2048}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusCropsize(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Focus Algorithm</td>
            <td style={tdStyle}>
              <select
                value={parameterValue.autoFocusAlgorithm || "LAPE"}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusAlgorithm(e.target.value))
                }
                style={{ width: "100%", padding: 5 }}
              >
                <option value="LAPE">LAPE (Laplacian)</option>
                <option value="GLVA">GLVA (Variance)</option>
                <option value="JPEG">JPEG (Compression)</option>
              </select>
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Static Offset (µm)</td>
            <td style={tdStyle}>
              <input
                type="number"
                step="0.1"
                min="-100"
                max="100"
                value={parameterValue.autoFocusStaticOffset || 0.0}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusStaticOffset(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Two-Stage Focus</td>
            <td style={tdStyle}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameterValue.autoFocusTwoStage || false}
                    onChange={(e) =>
                      dispatch(experimentSlice.setAutoFocusTwoStage(e.target.checked))
                    }
                  />
                }
                label=""
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Min Focus Position</td>
            <td style={tdStyle}>
              {/* No min/max, allow free entry, step=1 */}
              <input
                type="number"
                step="1"
                value={parameterValue.autoFocusMin}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusMin(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Max Focus Position</td>
            <td style={tdStyle}>
              {/* No min/max, allow free entry, step=1 */}
              <input
                type="number"
                step="1"
                value={parameterValue.autoFocusMax}
                onChange={(e) =>
                  dispatch(experimentSlice.setAutoFocusMax(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Step Size</td>
            <td style={tdStyle}>
              {/* No min/max, allow free entry, step=0.1 */}
              <input
                type="number"
                step="0.1"
                value={parameterValue.autoFocusStepSize}
                onChange={(e) =>
                  dispatch(
                    experimentSlice.setAutoFocusStepSize(Number(e.target.value))
                  )
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>

          {/* Z-stack */}
          <tr>
            <td rowSpan="3" style={tdStyle}>
              Z-Stack
            </td>
            <td style={tdStyle}>Min Focus Position</td>
            <td style={tdStyle}>
              {/* No min/max, allow free entry, step=1 */}
              <input
                type="number"
                step="1"
                value={parameterValue.zStackMin}
                onChange={(e) =>
                  dispatch(experimentSlice.setZStackMin(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Max Focus Position</td>
            <td style={tdStyle}>
              {/* No min/max, allow free entry, step=1 */}
              <input
                type="number"
                step="1"
                value={parameterValue.zStackMax}
                onChange={(e) =>
                  dispatch(experimentSlice.setZStackMax(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Step Size</td>
            <td style={tdStyle}>
              {/* No min/max, allow free entry, step=0.1 */}
              <input
                type="number"
                step="0.1"
                value={parameterValue.zStackStepSize}
                onChange={(e) =>
                  dispatch(experimentSlice.setZStackStepSize(Number(e.target.value)))
                }
                style={{ width: "100%", padding: 5 }}
              />
            </td>
          </tr>

          {/* Tile Overlap */}
          <tr>
            <td rowSpan="2" style={tdStyle}>
              Tile Overlap
            </td>
            <td style={tdStyle}>Width Overlap (%)</td>
            <td style={tdStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <input
                  type="range"
                  min="-50"
                  max="50"
                  step="5"
                  value={(parameterValue.overlapWidth * 100).toFixed(0)}
                  onChange={(e) =>
                    dispatch(experimentSlice.setOverlapWidth(Number(e.target.value) / 100))
                  }
                  style={{ flex: 1 }}
                />
                <span style={{ minWidth: "50px", textAlign: "right" }}>
                  {(parameterValue.overlapWidth * 100).toFixed(0)}%
                </span>
              </div>
              <small style={{ display: "block", color: "#666", fontSize: "0.8em", marginTop: "2px" }}>
                Negative = gap between tiles, Positive = overlap
              </small>
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Height Overlap (%)</td>
            <td style={tdStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <input
                  type="range"
                  min="-50"
                  max="50"
                  step="5"
                  value={(parameterValue.overlapHeight * 100).toFixed(0)}
                  onChange={(e) =>
                    dispatch(experimentSlice.setOverlapHeight(Number(e.target.value) / 100))
                  }
                  style={{ flex: 1 }}
                />
                <span style={{ minWidth: "50px", textAlign: "right" }}>
                  {(parameterValue.overlapHeight * 100).toFixed(0)}%
                </span>
              </div>
              <small style={{ display: "block", color: "#666", fontSize: "0.8em", marginTop: "2px" }}>
                Negative = gap between tiles, Positive = overlap
              </small>
            </td>
          </tr>

          {/* File Format Options */}
          <tr>
            <td rowSpan="4" style={tdStyle}>
              File Format
            </td>
            <td style={tdStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                OME-TIFF
                <Tooltip title="Creates OME-TIFF files with full metadata. Each image is stored as a separate TIFF file with comprehensive metadata for compatibility with ImageJ, FIJI, and other image analysis software." arrow>
                  <IconButton size="small" style={{ padding: "2px" }}>
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </div>
            </td>
            <td style={tdStyle}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameterValue.ome_write_tiff || false}
                    onChange={(e) =>
                      dispatch(experimentSlice.setOmeWriteTiff(e.target.checked))
                    }
                  />
                }
                label=""
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                OME-Zarr
                <Tooltip title="Creates OME-Zarr files for cloud-optimized storage. Zarr is a chunked, compressed array format that enables efficient access to large multidimensional datasets and is particularly suitable for remote access and parallel processing." arrow>
                  <IconButton size="small" style={{ padding: "2px" }}>
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </div>
            </td>
            <td style={tdStyle}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameterValue.ome_write_zarr || false}
                    onChange={(e) =>
                      dispatch(experimentSlice.setOmeWriteZarr(e.target.checked))
                    }
                  />
                }
                label=""
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                Stitched OME-TIFF
                <Tooltip title="Creates a single large stitched TIFF file by combining all tiles into one continuous image. This provides a complete overview of the entire scanned area but results in very large file sizes for big experiments." arrow>
                  <IconButton size="small" style={{ padding: "2px" }}>
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </div>
            </td>
            <td style={tdStyle}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameterValue.ome_write_stitched_tiff || false}
                    onChange={(e) =>
                      dispatch(experimentSlice.setOmeWriteStitchedTiff(e.target.checked))
                    }
                  />
                }
                label=""
              />
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                Individual TIFFs
                <Tooltip title="Saves each tile as a separate TIFF file with position-based naming (e.g., x5000_y3000_z1500_c0_i0042_p80.tif). This allows for easy access to individual tiles and is useful for distributed processing or when you need specific regions." arrow>
                  <IconButton size="small" style={{ padding: "2px" }}>
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </div>
            </td>
            <td style={tdStyle}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameterValue.ome_write_individual_tiffs || false}
                    onChange={(e) =>
                      dispatch(experimentSlice.setOmeWriteIndividualTiffs(e.target.checked))
                    }
                  />
                }
                label=""
              />
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default ParameterEditorComponent;
