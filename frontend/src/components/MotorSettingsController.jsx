/**
 * Motor Settings Controller Component
 * 
 * Provides a unified interface for configuring motor parameters per axis including:
 * - Motion settings (stepsize, speed, acceleration, backlash)
 * - Homing settings (direction, speed, endstop polarity, timeout)
 * - TMC driver settings (microsteps, current, stallguard)
 * - Global settings (axis order, CoreXY mode, enable settings)
 * 
 * Settings are read from the backend configuration and can be saved back
 * to both the device and the configuration file.
 */

import React, { useState, useEffect, useCallback } from "react";
import { useSelector } from "react-redux";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Tabs,
  Tab,
  TextField,
  FormControlLabel,
  Switch,
  Grid,
  Alert,
  CircularProgress,
  Divider,
  Tooltip,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
} from "@mui/material";
import {
  Settings,
  Save,
  Refresh,
  ExpandMore,
  Home,
  Speed,
  Tune,
  Memory,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  DirectionsRun,
} from "@mui/icons-material";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import createAxiosInstance from "../backendapi/createAxiosInstance";
import apiTMCSettingsGetForAxis from "../backendapi/apiTMCSettingsGetForAxis";
import apiMotorSettingsGet from "../backendapi/apiMotorSettingsGet";

// Tab panel component
const TabPanel = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`motor-tabpanel-${index}`}
    aria-labelledby={`motor-tab-${index}`}
    {...other}
  >
    {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
  </div>
);

// Axis configuration panel component
const AxisSettingsPanel = ({ axis, settings, onChange, onSave, isSaving, baseURL }) => {
  const [localSettings, setLocalSettings] = useState(settings || {});
  const [hasChanges, setHasChanges] = useState(false);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    setLocalSettings(settings || {});
    setHasChanges(false);
  }, [settings]);

  const handleChange = (section, field, value) => {
    setLocalSettings((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value,
      },
    }));
    setHasChanges(true);
  };

  const handleSave = () => {
    onSave(axis, localSettings);
    setHasChanges(false);
  };

  const motion = localSettings.motion || {};
  const homing = localSettings.homing || {};
  const limits = localSettings.limits || {};

  return (
    <Box>
      {/* Motion Settings */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Speed color="primary" />
            <Typography variant="subtitle1">Motion Settings</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Step Size (steps/µm)"
                type="number"
                size="small"
                value={motion.stepSize ?? 1}
                onChange={(e) =>
                  handleChange("motion", "stepSize", parseFloat(e.target.value))
                }
                helperText="Calibrated conversion factor"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Max Speed (steps/s)"
                type="number"
                size="small"
                value={motion.maxSpeed ?? 10000}
                onChange={(e) =>
                  handleChange("motion", "maxSpeed", parseInt(e.target.value))
                }
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Default Speed (steps/s)"
                type="number"
                size="small"
                value={motion.speed ?? 10000}
                onChange={(e) =>
                  handleChange("motion", "speed", parseInt(e.target.value))
                }
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Acceleration (steps/s²)"
                type="number"
                size="small"
                value={motion.acceleration ?? 1000000}
                onChange={(e) =>
                  handleChange("motion", "acceleration", parseInt(e.target.value))
                }
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Backlash (steps)"
                type="number"
                size="small"
                value={motion.backlash ?? 0}
                onChange={(e) =>
                  handleChange("motion", "backlash", parseInt(e.target.value))
                }
                helperText="Compensation for mechanical play"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Min Position"
                type="number"
                size="small"
                value={motion.minPos ?? ""}
                onChange={(e) =>
                  handleChange(
                    "motion",
                    "minPos",
                    e.target.value ? parseFloat(e.target.value) : null
                  )
                }
                helperText="Leave empty for no limit"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Max Position"
                type="number"
                size="small"
                value={motion.maxPos ?? ""}
                onChange={(e) =>
                  handleChange(
                    "motion",
                    "maxPos",
                    e.target.value ? parseFloat(e.target.value) : null
                  )
                }
                helperText="Leave empty for no limit"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Homing Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Home color="primary" />
            <Typography variant="subtitle1">Homing Settings</Typography>
            {homing.enabled && (
              <Chip label="Enabled" color="success" size="small" />
            )}
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={homing.enabled ?? false}
                    onChange={(e) =>
                      handleChange("homing", "enabled", e.target.checked)
                    }
                  />
                }
                label="Enable Homing (requires endstop)"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Homing Speed (steps/s)"
                type="number"
                size="small"
                value={homing.speed ?? 15000}
                onChange={(e) =>
                  handleChange("homing", "speed", parseInt(e.target.value))
                }
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Homing Direction</InputLabel>
                <Select
                  value={homing.direction ?? -1}
                  label="Homing Direction"
                  onChange={(e) =>
                    handleChange("homing", "direction", e.target.value)
                  }
                >
                  <MenuItem value={-1}>Negative (-1)</MenuItem>
                  <MenuItem value={1}>Positive (+1)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Endstop Polarity</InputLabel>
                <Select
                  value={homing.endstopPolarity ?? 1}
                  label="Endstop Polarity"
                  onChange={(e) =>
                    handleChange("homing", "endstopPolarity", e.target.value)
                  }
                >
                  <MenuItem value={0}>Normally Open (NO)</MenuItem>
                  <MenuItem value={1}>Normally Closed (NC)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Endstop Release (steps)"
                type="number"
                size="small"
                value={homing.endposRelease ?? 3000}
                onChange={(e) =>
                  handleChange("homing", "endposRelease", parseInt(e.target.value))
                }
                helperText="Back-off distance after hitting endstop"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Timeout (ms)"
                type="number"
                size="small"
                value={homing.timeout ?? 20000}
                onChange={(e) =>
                  handleChange("homing", "timeout", parseInt(e.target.value))
                }
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Home Steps (no endstop)"
                type="number"
                size="small"
                value={homing.homeSteps ?? 0}
                onChange={(e) =>
                  handleChange("homing", "homeSteps", parseInt(e.target.value))
                }
                helperText="Steps to move if no endstop present (0 = disabled)"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={homing.homeOnStart ?? false}
                    onChange={(e) =>
                      handleChange("homing", "homeOnStart", e.target.checked)
                    }
                  />
                }
                label="Home on Startup"
              />
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Limit Settings */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Warning color="primary" />
            <Typography variant="subtitle1">Limit Settings</Typography>
            {(limits.enabled || limits.hardLimitsEnabled) && (
              <Chip label="Active" color="success" size="small" />
            )}
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={limits.enabled ?? false}
                    onChange={(e) =>
                      handleChange("limits", "enabled", e.target.checked)
                    }
                  />
                }
                label="Enable Position Limits (software - prevent movement below 0)"
              />
            </Grid>
            <Grid item xs={12}>
              <Divider sx={{ my: 1 }} />
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                Hardware Hard Limits use physical endstop switches to prevent movement beyond safe range
              </Typography>
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={limits.hardLimitsEnabled ?? false}
                    onChange={(e) =>
                      handleChange("limits", "hardLimitsEnabled", e.target.checked)
                    }
                  />
                }
                label="Enable Hardware Hard Limits (physical endstops)"
              />
            </Grid>
            {limits.hardLimitsEnabled && (
              <Grid item xs={12}>
                <Alert severity="info" sx={{ mt: 1 }}>
                  Hardware hard limits enabled. Motor will stop immediately when endstop is triggered.
                  Make sure physical endstops are properly connected to the motor driver.
                </Alert>
              </Grid>
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* Test Movement Buttons */}
      <Box sx={{ mt: 2, p: 2, bgcolor: "action.hover", borderRadius: 1 }}>
        <Typography variant="subtitle2" gutterBottom>
          Test Movement
        </Typography>
        <Box sx={{ display: "flex", gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            disabled={testing}
            onClick={async () => {
              setTesting(true);
              try {
                const axios = createAxiosInstance(baseURL);
                await axios.get('/PositionerController/stepPositionerDown', {
                  params: { axis, positionerName: 'ESP32Stage' }
                });
                // Move 100 steps
                await axios.get('/PositionerController/move', {
                  positionerName: 'ESP32Stage',
                  axis: axis,
                  dist: -100,
                  isAbsolute: false,
                  isBlocking: false
                });
              } catch (e) {
                console.error('Test movement failed:', e);
              }
              setTesting(false);
            }}
          >
            -100 steps
          </Button>
          <Button
            variant="outlined"
            size="small"
            disabled={testing}
            onClick={async () => {
              setTesting(true);
              try {
                const axios = createAxiosInstance(baseURL);
                await axios.get('/PositionerController/move', {
                  positionerName: 'ESP32Stage',
                  axis: axis,
                  dist: 100,
                  isAbsolute: false,
                  isBlocking: false
                });
              } catch (e) {
                console.error('Test movement failed:', e);
              }
              setTesting(false);
            }}
          >
            +100 steps
          </Button>
        </Box>
      </Box>

      {/* Save Button */}
      <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end", gap: 1 }}>
        {hasChanges && (
          <Chip
            label="Unsaved changes"
            color="warning"
            size="small"
            sx={{ mr: 1 }}
          />
        )}
        <Button
          variant="contained"
          startIcon={isSaving ? <CircularProgress size={20} /> : <Save />}
          onClick={handleSave}
          disabled={!hasChanges || isSaving}
        >
          Save {axis} Settings
        </Button>
      </Box>
    </Box>
  );
};

// TMC Settings Panel component
const TMCSettingsPanel = ({ axis, onSave, isSaving, baseURL, currentStepSize, onStepSizeChange }) => {
  const [settings, setSettings] = useState({
    msteps: 16,
    rmsCurrent: 500,
    sgthrs: 10,
    semin: 5,
    semax: 2,
    blankTime: 24,
    toff: 3,
  });
  const [loading, setLoading] = useState(true);
  const [previousMsteps, setPreviousMsteps] = useState(16);

  // Load TMC settings from device on mount
  useEffect(() => {
    const loadTMCSettings = async () => {
      try {
        const result = await apiTMCSettingsGetForAxis(baseURL, axis);
        if (result.success && result.settings) {
          setSettings(result.settings);
          setPreviousMsteps(result.settings.msteps || 16);
        }
      } catch (error) {
        console.error('Error loading TMC settings:', error);
      } finally {
        setLoading(false);
      }
    };
    loadTMCSettings();
  }, [axis, baseURL]);

  const handleChange = (field, value) => {
    // Auto-scale stepsize when microsteps change
    if (field === 'msteps' && currentStepSize && onStepSizeChange) {
      const newMsteps = parseInt(value);
      const oldMsteps = previousMsteps;
      if (newMsteps !== oldMsteps && oldMsteps > 0) {
        // Scale stepsize to maintain physical movement distance
        // If microsteps increase, stepsize should increase proportionally
        const newStepSize = currentStepSize * (newMsteps / oldMsteps);
        onStepSizeChange(axis, newStepSize);
      }
      setPreviousMsteps(newMsteps);
    }
    setSettings((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <Box>
      <Alert severity="info" sx={{ mb: 2 }}>
        TMC settings are sent directly to the stepper driver. Changes take effect immediately.
      </Alert>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Microsteps</InputLabel>
            <Select
              value={settings.msteps}
              label="Microsteps"
              onChange={(e) => handleChange("msteps", e.target.value)}
            >
              {[1, 2, 4, 8, 16, 32, 64, 128, 256].map((val) => (
                <MenuItem key={val} value={val}>
                  {val}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            label="RMS Current (mA)"
            type="number"
            size="small"
            value={settings.rmsCurrent}
            onChange={(e) =>
              handleChange("rmsCurrent", parseInt(e.target.value))
            }
            helperText="Motor current (200-2000 mA)"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            label="StallGuard Threshold"
            type="number"
            size="small"
            value={settings.sgthrs}
            onChange={(e) => handleChange("sgthrs", parseInt(e.target.value))}
            helperText="Stall detection sensitivity"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            label="SE Min (CoolStep)"
            type="number"
            size="small"
            value={settings.semin}
            onChange={(e) => handleChange("semin", parseInt(e.target.value))}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            label="SE Max (CoolStep)"
            type="number"
            size="small"
            value={settings.semax}
            onChange={(e) => handleChange("semax", parseInt(e.target.value))}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            label="Blank Time"
            type="number"
            size="small"
            value={settings.blankTime}
            onChange={(e) =>
              handleChange("blankTime", parseInt(e.target.value))
            }
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <TextField
            fullWidth
            label="T-Off"
            type="number"
            size="small"
            value={settings.toff}
            onChange={(e) => handleChange("toff", parseInt(e.target.value))}
          />
        </Grid>
      </Grid>
      <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end" }}>
        <Button
          variant="contained"
          color="secondary"
          startIcon={isSaving ? <CircularProgress size={20} /> : <Memory />}
          onClick={() => onSave(axis, settings)}
          disabled={isSaving}
        >
          Apply TMC Settings to {axis}
        </Button>
      </Box>
    </Box>
  );
};

// Global Settings Panel component
const GlobalSettingsPanel = ({ settings, onChange, onSave, isSaving }) => {
  const [localSettings, setLocalSettings] = useState(settings || {});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setLocalSettings(settings || {});
    setHasChanges(false);
  }, [settings]);

  const handleChange = (field, value) => {
    setLocalSettings((prev) => ({
      ...prev,
      [field]: value,
    }));
    setHasChanges(true);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Global Motor Settings
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <FormControlLabel
            control={
              <Switch
                checked={localSettings.isEnabled ?? true}
                onChange={(e) => handleChange("isEnabled", e.target.checked)}
              />
            }
            label="Motors Enabled"
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControlLabel
            control={
              <Switch
                checked={localSettings.enableAuto ?? true}
                onChange={(e) => handleChange("enableAuto", e.target.checked)}
              />
            }
            label="Auto-Enable (auto power off when idle)"
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControlLabel
            control={
              <Switch
                checked={localSettings.isCoreXY ?? false}
                onChange={(e) => handleChange("isCoreXY", e.target.checked)}
              />
            }
            label="CoreXY Geometry"
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControlLabel
            control={
              <Switch
                checked={localSettings.isDualAxis ?? false}
                onChange={(e) => handleChange("isDualAxis", e.target.checked)}
              />
            }
            label="Dual Axis Mode (A+Z linked)"
          />
        </Grid>
        <Grid item xs={12}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Axis Order (A=0, X=1, Y=2, Z=3)
          </Typography>
          <Box sx={{ display: "flex", gap: 1 }}>
            {(localSettings.axisOrder || [0, 1, 2, 3]).map((val, idx) => (
              <TextField
                key={idx}
                label={["A", "X", "Y", "Z"][idx]}
                type="number"
                size="small"
                value={val}
                onChange={(e) => {
                  const newOrder = [...(localSettings.axisOrder || [0, 1, 2, 3])];
                  newOrder[idx] = parseInt(e.target.value);
                  handleChange("axisOrder", newOrder);
                }}
                sx={{ width: 70 }}
              />
            ))}
          </Box>
        </Grid>
      </Grid>
      <Box sx={{ mt: 2, display: "flex", justifyContent: "flex-end", gap: 1 }}>
        {hasChanges && (
          <Chip
            label="Unsaved changes"
            color="warning"
            size="small"
            sx={{ mr: 1 }}
          />
        )}
        <Button
          variant="contained"
          startIcon={isSaving ? <CircularProgress size={20} /> : <Save />}
          onClick={() => onSave(localSettings)}
          disabled={!hasChanges || isSaving}
        >
          Save Global Settings
        </Button>
      </Box>
    </Box>
  );
};

// Main component
const MotorSettingsController = () => {
  const [tabIndex, setTabIndex] = useState(0);
  const [tmcAxisIndex, setTmcAxisIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [motorSettings, setMotorSettings] = useState(null);

  // Get connection state
  const uc2State = useSelector(uc2Slice.getUc2State);
  const isBackendConnected = uc2State.backendConnected;
  const connectionSettings = useSelector(getConnectionSettingsState);
  const baseURL = connectionSettings.baseURL || "http://localhost:8001";

  // Fetch motor settings from backend
  const fetchSettings = useCallback(async () => {
    if (!isBackendConnected) return;

    setLoading(true);
    setError(null);

    try {
      const api = createAxiosInstance();
      const response = await api.get("/UC2ConfigController/getMotorSettings");
      
      if (response.data.error) {
        setError(response.data.error);
      } else {
        setMotorSettings(response.data);
      }
    } catch (err) {
      console.error("Error fetching motor settings:", err);
      setError("Failed to fetch motor settings: " + (err.message || "Unknown error"));
    } finally {
      setLoading(false);
    }
  }, [isBackendConnected]);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  // Save axis settings
  const handleSaveAxisSettings = async (axis, settings) => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const api = createAxiosInstance();
      const response = await api.post(
        `/UC2ConfigController/setMotorSettingsForAxis?axis=${axis}`,
        settings
      );

      if (response.data.success) {
        setSuccess(`Settings for ${axis} axis saved successfully`);
        // Update local state
        setMotorSettings((prev) => ({
          ...prev,
          axes: {
            ...prev?.axes,
            [axis]: settings,
          },
        }));
      } else {
        setError(response.data.error || "Failed to save settings");
      }
    } catch (err) {
      console.error("Error saving axis settings:", err);
      setError("Failed to save settings: " + (err.message || "Unknown error"));
    } finally {
      setSaving(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };

  // Save TMC settings
  const handleSaveTMCSettings = async (axis, settings) => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const api = createAxiosInstance();
      const response = await api.post(
        `/UC2ConfigController/setTMCSettingsForAxis?axis=${axis}`,
        settings
      );

      if (response.data.success) {
        setSuccess(`TMC settings for ${axis} axis applied successfully`);
      } else {
        setError(response.data.error || "Failed to apply TMC settings");
      }
    } catch (err) {
      console.error("Error saving TMC settings:", err);
      setError("Failed to apply TMC settings: " + (err.message || "Unknown error"));
    } finally {
      setSaving(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };

  // Handle stepsize change when microsteps change
  const handleStepSizeChange = async (axis, newStepSize) => {
    try {
      // Update the motorSettings state with new stepsize
      setMotorSettings((prev) => ({
        ...prev,
        axes: {
          ...prev?.axes,
          [axis]: {
            ...prev?.axes?.[axis],
            motion: {
              ...prev?.axes?.[axis]?.motion,
              stepSize: newStepSize,
            },
          },
        },
      }));
    } catch (err) {
      console.error("Error updating stepsize:", err);
    }
  };

  // Save global settings
  const handleSaveGlobalSettings = async (settings) => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const api = createAxiosInstance();
      const response = await api.post(
        "/UC2ConfigController/setGlobalMotorSettings",
        settings
      );

      if (response.data.success) {
        setSuccess("Global settings saved successfully");
        setMotorSettings((prev) => ({
          ...prev,
          global: settings,
        }));
      } else {
        setError(response.data.error || "Failed to save global settings");
      }
    } catch (err) {
      console.error("Error saving global settings:", err);
      setError("Failed to save settings: " + (err.message || "Unknown error"));
    } finally {
      setSaving(false);
      setTimeout(() => setSuccess(null), 3000);
    }
  };

  // Tabs for axes
  const axisTabs = ["X", "Y", "Z", "A", "Global", "TMC"];

  if (!isBackendConnected) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">
          Backend connection required. Please connect to ImSwitch first.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", mb: 3, gap: 2 }}>
        <Settings sx={{ fontSize: 32, color: "primary.main" }} />
        <Box>
          <Typography variant="h4">Motor Settings</Typography>
          <Typography variant="body2" color="text.secondary">
            Configure motor parameters, homing, and TMC driver settings per axis
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        <Tooltip title="Refresh settings from device">
          <IconButton onClick={fetchSettings} disabled={loading}>
            <Refresh />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Status Messages */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* Loading State */}
      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Card>
          <CardContent>
            {/* Tabs */}
            <Tabs
              value={tabIndex}
              onChange={(e, newValue) => setTabIndex(newValue)}
              variant="scrollable"
              scrollButtons="auto"
            >
              {axisTabs.map((axis, idx) => (
                <Tab
                  key={axis}
                  label={axis}
                  icon={
                    axis === "Global" ? (
                      <Tune />
                    ) : axis === "TMC" ? (
                      <Memory />
                    ) : (
                      <DirectionsRun />
                    )
                  }
                  iconPosition="start"
                />
              ))}
            </Tabs>

            <Divider sx={{ my: 2 }} />

            {/* Tab Panels for each axis */}
            {["X", "Y", "Z", "A"].map((axis, idx) => (
              <TabPanel key={axis} value={tabIndex} index={idx}>
                <AxisSettingsPanel
                  axis={axis}
                  settings={motorSettings?.axes?.[axis]}
                  onSave={handleSaveAxisSettings}
                  isSaving={saving}
                  baseURL={baseURL}
                />
              </TabPanel>
            ))}

            {/* Global Settings Tab */}
            <TabPanel value={tabIndex} index={4}>
              <GlobalSettingsPanel
                settings={motorSettings?.global}
                onSave={handleSaveGlobalSettings}
                isSaving={saving}
              />
            </TabPanel>

            {/* TMC Settings Tab */}
            <TabPanel value={tabIndex} index={5}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Select an axis to configure its TMC stepper driver settings. 
                Note: Changing microsteps will automatically scale the stepsize to maintain physical movement distance.
              </Typography>
              <Tabs
                value={tmcAxisIndex}
                onChange={(e, newValue) => setTmcAxisIndex(newValue)}
                variant="scrollable"
                scrollButtons="auto"
                sx={{ mb: 2 }}
              >
                {["X", "Y", "Z", "A"].map((axis) => (
                  <Tab key={axis} label={axis} />
                ))}
              </Tabs>
              {["X", "Y", "Z", "A"].map((axis, idx) => (
                <Box key={axis} hidden={tmcAxisIndex !== idx}>
                  {tmcAxisIndex === idx && (
                    <TMCSettingsPanel
                      axis={axis}
                      onSave={handleSaveTMCSettings}
                      isSaving={saving}
                      baseURL={baseURL}
                      currentStepSize={motorSettings?.axes?.[axis]?.motion?.stepSize}
                      onStepSizeChange={handleStepSizeChange}
                    />
                  )}
                </Box>
              ))}
            </TabPanel>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default MotorSettingsController;
