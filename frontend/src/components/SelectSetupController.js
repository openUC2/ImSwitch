import React, { useEffect, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Paper,
  Box,
  Typography,
  Button,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Grid,
  FormControlLabel,
  Switch,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  LinearProgress,
} from "@mui/material";
import {
  Refresh as RefreshIcon,
} from "@mui/icons-material";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import {
  setNotification,
  clearNotification,
} from "../state/slices/NotificationSlice";

const SelectSetupController = ({ hostIP, hostPort }) => {
  // Redux dispatcher
  const dispatch = useDispatch();

  // Access global Redux state
  const uc2State = useSelector(uc2Slice.getUc2State);

  // Use Redux state instead of local useState
  const availableSetups = uc2State.availableSetups;
  const selectedSetup = uc2State.selectedSetup;
  const isDialogOpen = uc2State.isDialogOpen;
  const restartSoftware = uc2State.restartSoftware;
  const currentActiveFilename = uc2State.currentActiveFilename;
  const isLoadingCurrentFilename = uc2State.isLoadingCurrentFilename;
  const isRestarting = uc2State.isRestarting;

  const fetchCurrentActiveFilename = useCallback(() => {
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/getCurrentSetupFilename`;
    dispatch(uc2Slice.setIsLoadingCurrentFilename(true));
    
    fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        // Handle different possible response formats
        let filename = null;
        
        if (typeof data === 'string') {
          filename = data;
        } else if (data && typeof data === 'object') {
          filename = data.current_setup || data.currentSetupFilename || data.setupFileName || JSON.stringify(data);
        } else {
          filename = String(data);
        }
        
        // Extract just the filename from full path if needed
        if (filename && filename.includes('/')) {
          filename = filename.split('/').pop();
        } else if (filename && filename.includes('\\')) {
          // Handle Windows paths
          filename = filename.split('\\').pop();
        }
        
        const activeFilename = filename || 'current_config.json';
        dispatch(uc2Slice.setCurrentActiveFilename(activeFilename));
      })
      .catch((error) => {
        console.error("Error fetching current setup filename:", error);
        dispatch(uc2Slice.setCurrentActiveFilename('current_config.json')); // fallback
      })
      .finally(() => {
        dispatch(uc2Slice.setIsLoadingCurrentFilename(false));
      });
  }, [hostIP, hostPort, dispatch]);

  const fetchAvailableSetups = useCallback(() => {
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/returnAvailableSetups`;
    dispatch(
      setNotification({
        message: "Loading available setups...",
        type: "info",
      })
    );

    fetch(url)
      .then((response) => response.json())
      .then((data) => {
        dispatch(uc2Slice.setAvailableSetups(data.available_setups || []));
        dispatch(
          setNotification({
            message: `Found ${
              data.available_setups?.length || 0
            } configuration files`,
            type: "success",
          })
        );
        setTimeout(() => dispatch(clearNotification()), 3000);
      })
      .catch((error) => {
        console.error("Error fetching setups:", error);
        dispatch(
          setNotification({
            message: "Failed to load configuration files",
            type: "error",
          })
        );
        setTimeout(() => dispatch(clearNotification()), 3000);
      });
  }, [hostIP, hostPort, dispatch]);

  const monitorRestartStatus = useCallback(() => {
    let retryCount = 0;
    const maxRetries = 30; // 5 minutes with 10-second intervals

    const checkStatus = () => {
      fetch(`${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/is_connected`)
        .then((res) => res.json())
        .then((data) => {
          if (data === true) {
            dispatch(
              setNotification({
                message: "ImSwitch is back online!",
                type: "success",
              })
            );
            dispatch(uc2Slice.setIsRestarting(false));
            // Refresh current active filename after restart
            fetchCurrentActiveFilename();
            setTimeout(() => dispatch(clearNotification()), 3000);
          } else {
            throw new Error("Not connected yet");
          }
        })
        .catch(() => {
          retryCount++;
          if (retryCount < maxRetries) {
            dispatch(
              setNotification({
                message: `Waiting for ImSwitch to restart... (${retryCount}/${maxRetries})`,
                type: "info",
              })
            );
            setTimeout(checkStatus, 10000); // Check every 10 seconds
          } else {
            dispatch(
              setNotification({
                message:
                  "ImSwitch restart taking longer than expected. Please check manually.",
                type: "warning",
              })
            );
            dispatch(uc2Slice.setIsRestarting(false));
          }
        });
    };

    // Start checking after 5 seconds
    setTimeout(checkStatus, 5000);
  }, [hostIP, hostPort, dispatch, fetchCurrentActiveFilename]);

  useEffect(() => {
    fetchAvailableSetups();
    fetchCurrentActiveFilename();
  }, [fetchAvailableSetups, fetchCurrentActiveFilename]);

  // Auto-select the current active file when both are available
  useEffect(() => {
    if (currentActiveFilename && availableSetups.includes(currentActiveFilename) && !selectedSetup) {
      dispatch(uc2Slice.setSelectedSetup(currentActiveFilename));
    }
  }, [currentActiveFilename, availableSetups, selectedSetup, dispatch]);

  const handleSetupChange = (event) => {
    dispatch(uc2Slice.setSelectedSetup(event.target.value));
  };

  const handleSetSetup = () => {
    if (!selectedSetup) {
      dispatch(
        setNotification({
          message: "Please select a setup before proceeding",
          type: "warning",
        })
      );
      return;
    }

    dispatch(uc2Slice.setIsRestarting(true));
    dispatch(
      setNotification({
        message: "Setting up configuration...",
        type: "info",
      })
    );

    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/setSetupFileName?setupFileName=${encodeURIComponent(
      selectedSetup
    )}&restartSoftware=${restartSoftware}`;

    fetch(url, { method: "GET" })
      .then((response) => response.json())
      .then((data) => {
        console.log("Setup selected:", data);
        dispatch(uc2Slice.setIsDialogOpen(true));

        if (restartSoftware) {
          dispatch(
            setNotification({
              message: "ImSwitch is restarting... Please wait",
              type: "info",
            })
          );
          // Start monitoring connection status
          monitorRestartStatus();
        } else {
          dispatch(
            setNotification({
              message: "Configuration updated successfully",
              type: "success",
            })
          );
          dispatch(uc2Slice.setIsRestarting(false));
          // Refresh current active filename after successful change
          fetchCurrentActiveFilename();
          setTimeout(() => dispatch(clearNotification()), 3000);
        }
      })
      .catch((error) => {
        console.error("Error setting setup:", error);
        if (restartSoftware) {
          dispatch(
            setNotification({
              message: "Configuration change initiated. ImSwitch is restarting... Please wait",
              type: "info",
            })
          );
          // Start monitoring connection status even on "error" as it might just be a restart
          monitorRestartStatus();
        } else {
          dispatch(
            setNotification({
              message: "There was an issue applying the configuration. Please try again or check the connection.",
              type: "warning",
            })
          );
          dispatch(uc2Slice.setIsRestarting(false));
        }
      });
  };

  const handleDialogClose = () => {
    dispatch(uc2Slice.setIsDialogOpen(false)); // Close the dialog
  };

  return (
    <Paper>
      <Box p={3}>
        <Typography variant="h4" gutterBottom>
          Select Setup Configuration
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}
            >
              <Typography variant="h6">Select Available Setup</Typography>
              <Button
                variant="outlined"
                size="small"
                startIcon={<RefreshIcon />}
                onClick={() => {
                  fetchAvailableSetups();
                  fetchCurrentActiveFilename();
                }}
                disabled={isRestarting}
              >
                Refresh
              </Button>
            </Box>

            {/* Current Active Configuration Display */}
            <Paper 
              elevation={1} 
              sx={{ 
                p: 2, 
                mb: 3, 
                backgroundColor: "info.light",
                color: "info.contrastText"
              }}
            >
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                <Typography variant="subtitle1" fontWeight="bold">
                  Currently Active Configuration:
                </Typography>
                {isLoadingCurrentFilename && (
                  <CircularProgress size={16} color="inherit" />
                )}
              </Box>
              <Typography 
                variant="body1" 
                sx={{ 
                  fontFamily: "monospace",
                  fontWeight: "bold"
                }}
              >
                {currentActiveFilename || "Loading..."}
              </Typography>
            </Paper>

            {isRestarting && (
              <Box sx={{ mb: 2 }}>
                <LinearProgress />
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mt: 1 }}
                >
                  {restartSoftware
                    ? "ImSwitch is restarting... Please wait, this may take a few minutes."
                    : "Processing configuration change..."}
                </Typography>
              </Box>
            )}

            <FormControl fullWidth style={{ marginBottom: "20px" }}>
              <InputLabel id="setup-select-label">
                Available Setups
              </InputLabel>
              <Select
                labelId="setup-select-label"
                value={selectedSetup}
                onChange={handleSetupChange}
                disabled={isRestarting}
              >
                {availableSetups.map((setup, index) => (
                  <MenuItem 
                    key={index} 
                    value={setup}
                    sx={{
                      backgroundColor: setup === currentActiveFilename ? "success.light" : "inherit",
                      color: setup === currentActiveFilename ? "success.contrastText" : "inherit",
                      fontWeight: setup === currentActiveFilename ? "bold" : "normal",
                      "&:hover": {
                        backgroundColor: setup === currentActiveFilename ? "success.main" : "action.hover",
                      }
                    }}
                  >
                    {setup}
                    {setup === currentActiveFilename && " ⭐ (Currently Active)"}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Switch to enable/disable restart */}
            <FormControlLabel
              control={
                <Switch
                  checked={restartSoftware}
                  onChange={(e) =>
                    dispatch(uc2Slice.setRestartSoftware(e.target.checked))
                  }
                  disabled={isRestarting}
                />
              }
              label="Restart Software After Setup"
            />

            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                onClick={handleSetSetup}
                disabled={!selectedSetup || isRestarting}
                startIcon={
                  isRestarting ? <CircularProgress size={20} /> : null
                }
              >
                {isRestarting ? "Processing..." : "Apply Setup"}
              </Button>
            </Box>

            {/* Confirmation Dialog */}
            <Dialog open={isDialogOpen} onClose={handleDialogClose}>
              <DialogTitle>
                {restartSoftware ? "System Restart in Progress" : "Configuration Updated"}
              </DialogTitle>
              <DialogContent>
                <Typography>
                  {restartSoftware ? (
                    <>
                      ImSwitch is restarting with the new configuration. This process may take several minutes.
                      <br/><br/>
                      Please wait patiently while the system:
                      <br/>• Applies the new configuration
                      <br/>• Restarts the software
                      <br/>• Reconnects to hardware
                      <br/><br/>
                      You will receive a notification when the system is ready.
                    </>
                  ) : (
                    "The configuration has been updated successfully without restarting the software."
                  )}
                </Typography>
              </DialogContent>
              <DialogActions>
                <Button onClick={handleDialogClose} color="primary">
                  {restartSoftware ? "Understood" : "OK"}
                </Button>
              </DialogActions>
            </Dialog>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default SelectSetupController;
