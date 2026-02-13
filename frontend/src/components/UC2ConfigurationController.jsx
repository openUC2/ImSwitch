import {
  Preview as PreviewIcon,
  Refresh as RefreshIcon,
  Save as SaveIcon,
  AutoFixHigh as WizardIcon,
  CheckCircle,
  ErrorOutline,
  SettingsApplications,
  Build,
} from "@mui/icons-material";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Switch,
  Tab,
  Tabs,
  TextField,
  Typography,
  Chip,
} from "@mui/material";
import AceEditor from "react-ace";
import "ace-builds/src-noconflict/mode-json";
import "ace-builds/src-noconflict/theme-github";
import { JsonEditor } from "json-edit-react";
import React, { useCallback, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  clearNotification,
  setNotification,
} from "../state/slices/NotificationSlice";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import {
  createConfigurationPreview,
  validateConfiguration,
  validateJsonString,
  validateFileName,
} from "../utils/configValidation";
import ConfigurationPreviewDialog from "./ConfigurationPreviewDialog";
import ConfigurationWizard from "./ConfigurationWizard";

const TabPanel = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    {...other}
    id={`simple-tabpanel-${index}`}
    aria-labelledby={`simple-tab-${index}`}
  >
    {value === index && <Box p={3}>{children}</Box>}
  </div>
);

const UC2ConfigurationController = () => {
  // Redux dispatcher
  const dispatch = useDispatch();

  // Get connection settings from Redux state
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip || "http://localhost";
  const hostPort = connectionSettings.apiPort || "8000";

  const [showConfigWizard, setShowConfigWizard] = React.useState(false);
  const hasAutoSelectedRef = React.useRef(false);

  // Access global Redux state
  const uc2State = useSelector(uc2Slice.getUc2State);

  // Use Redux state instead of local useState
  const availableSetups = uc2State.availableSetups;
  const tabIndex = uc2State.tabIndex;
  const selectedSetup = uc2State.selectedSetup;
  const isDialogOpen = uc2State.isDialogOpen;
  const restartSoftware = uc2State.restartSoftware;
  const uc2Connected = uc2State.uc2Connected; // Hardware connected
  const selectedFileForEdit = uc2State.selectedFileForEdit;
  const editorJson = uc2State.editorJson;
  const editorJsonText = uc2State.editorJsonText;
  const useAceEditor = uc2State.useAceEditor;
  const newFileName = uc2State.newFileName;
  const setAsCurrentConfig = uc2State.setAsCurrentConfig;
  const restartAfterSave = uc2State.restartAfterSave;
  const overwriteFile = uc2State.overwriteFile;
  const currentActiveFilename = uc2State.currentActiveFilename;

  // New state properties
  const isLoadingFile = uc2State.isLoadingFile;
  const isSavingFile = uc2State.isSavingFile;
  const isRestarting = uc2State.isRestarting;
  const validationResult = uc2State.validationResult;
  const configPreview = uc2State.configPreview;
  const showPreviewDialog = uc2State.showPreviewDialog;

  const fetchAvailableSetups = useCallback(() => {
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/returnAvailableSetups`;
    dispatch(
      setNotification({
        message: "Loading available setups...",
        type: "info",
      }),
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
          }),
        );
        setTimeout(() => dispatch(clearNotification()), 3000);
      })
      .catch((error) => {
        console.error("Error fetching setups:", error);
        dispatch(
          setNotification({
            message: "Failed to load configuration files",
            type: "error",
          }),
        );
        setTimeout(() => dispatch(clearNotification()), 3000);
      });
  }, [hostIP, hostPort, dispatch]);

  const fetchCurrentSetup = useCallback(() => {
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/getCurrentSetupFilename`;
    dispatch(uc2Slice.setIsLoadingCurrentFilename(true));
    fetch(url)
      .then((response) => response.json())
      .then((data) => {
        const currentSetup = data.current_setup || "";
        const setupName = currentSetup.split(/[\\/]/).pop();
        const normalized = setupName || currentSetup;
        dispatch(uc2Slice.setCurrentActiveFilename(normalized));

        if (
          !hasAutoSelectedRef.current &&
          normalized &&
          availableSetups.includes(normalized)
        ) {
          dispatch(uc2Slice.setSelectedSetup(normalized));
          hasAutoSelectedRef.current = true;
        }
      })
      .catch((error) => {
        console.error("Error fetching current setup:", error);
        dispatch(
          setNotification({
            message: "Failed to load current configuration",
            type: "warning",
          }),
        );
        setTimeout(() => dispatch(clearNotification()), 3000);
      })
      .finally(() => {
        dispatch(uc2Slice.setIsLoadingCurrentFilename(false));
      });
  }, [hostIP, hostPort, dispatch, availableSetups]);

  const monitorRestartStatus = useCallback(() => {
    let retryCount = 0;
    const maxRetries = 30; // 5 minutes with 10-second intervals

    const checkStatus = () => {
      fetch(`${hostIP}:${hostPort}/imswitch/api/version`)
        .then((res) => {
          if (!res.ok) {
            throw new Error("Backend not ready");
          }
          return res.json().catch(() => ({}));
        })
        .then(() => {
          dispatch(
            setNotification({
              message: "ImSwitch backend is back online!",
              type: "success",
            }),
          );
          dispatch(uc2Slice.setIsRestarting(false));
          setTimeout(() => dispatch(clearNotification()), 3000);
          fetchAvailableSetups();
          fetchCurrentSetup();
        })
        .catch(() => {
          retryCount++;
          if (retryCount < maxRetries) {
            dispatch(
              setNotification({
                message: `Waiting for ImSwitch to restart... (${retryCount}/${maxRetries})`,
                type: "info",
              }),
            );
            setTimeout(checkStatus, 10000); // Check every 10 seconds
          } else {
            dispatch(
              setNotification({
                message:
                  "ImSwitch restart taking longer than expected. Please check manually.",
                type: "warning",
              }),
            );
            dispatch(uc2Slice.setIsRestarting(false));
          }
        });
    };

    // Start checking after 5 seconds
    setTimeout(checkStatus, 10000);
  }, [hostIP, hostPort, dispatch]);

  useEffect(() => {
    fetchAvailableSetups();
  }, [dispatch, fetchAvailableSetups]);

  useEffect(() => {
    if (availableSetups.length > 0) {
      fetchCurrentSetup();
    }
  }, [availableSetups]);

  // Note: Connection monitoring is now handled centrally by WebSocketHandler
  // This eliminates duplicate API calls and potential conflicts with ConnectionSettings testing

  const handleTabChange = (event, newValue) => {
    dispatch(uc2Slice.setTabIndex(newValue));
  };

  const handleSetupChange = (event) => {
    dispatch(uc2Slice.setSelectedSetup(event.target.value));
  };

  const handleSetSetup = () => {
    if (!selectedSetup) {
      dispatch(
        setNotification({
          message: "Please select a setup before proceeding",
          type: "warning",
        }),
      );
      return;
    }

    dispatch(uc2Slice.setIsRestarting(true));
    dispatch(
      setNotification({
        message: "Setting up configuration...",
        type: "info",
      }),
    );

    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/setSetupFileName?setupFileName=${encodeURIComponent(
      selectedSetup,
    )}&restartSoftware=${restartSoftware}`;

    fetch(url, { method: "GET" })
      .then(async (response) => {
        const text = await response.text();
        let data = null;
        try {
          data = JSON.parse(text);
        } catch (error) {
          data = text;
        }
        if (!response.ok) {
          throw new Error(`Failed to set configuration: ${response.status}`);
        }
        return data;
      })
      .then((data) => {
        console.log("Setup selected:", data);
        dispatch(uc2Slice.setIsDialogOpen(true));

        if (restartSoftware) {
          dispatch(
            setNotification({
              message: "ImSwitch is restarting... Please wait",
              type: "info",
            }),
          );
          // Start monitoring connection status
          monitorRestartStatus();
        } else {
          dispatch(
            setNotification({
              message: "Configuration updated successfully",
              type: "success",
            }),
          );
          dispatch(uc2Slice.setIsRestarting(false));
          setTimeout(() => dispatch(clearNotification()), 3000);
        }
      })
      .catch((error) => {
        console.error("Error setting setup:", error);
        if (restartSoftware) {
          dispatch(
            setNotification({
              message: "ImSwitch is restarting... Please wait",
              type: "info",
            }),
          );
          monitorRestartStatus();
          return;
        }
        dispatch(
          setNotification({
            message: "Failed to set configuration",
            type: "error",
          }),
        );
        dispatch(uc2Slice.setIsRestarting(false));
      });
  };

  const handleDialogClose = () => {
    dispatch(uc2Slice.setIsDialogOpen(false)); // Close the dialog
  };

  const handleNewConfig = () => {
    dispatch(uc2Slice.setUseAceEditor(true));
    dispatch(uc2Slice.setEditorJson(null));
    dispatch(uc2Slice.setEditorJsonText("{\n  \n}"));
    dispatch(uc2Slice.setSelectedFileForEdit(""));
  };

  const handleLoadSetupFile = () => {
    if (!selectedFileForEdit) {
      dispatch(
        setNotification({
          message: "Please select a file to load",
          type: "warning",
        }),
      );
      return;
    }

    dispatch(uc2Slice.setIsLoadingFile(true));
    dispatch(uc2Slice.setUseAceEditor(false));
    dispatch(
      setNotification({
        message: "Loading configuration file...",
        type: "info",
      }),
    );

    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/readSetupFile?setupFileName=${encodeURIComponent(
      selectedFileForEdit,
    )}`;

    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        dispatch(uc2Slice.setEditorJson(data));
        dispatch(
          setNotification({
            message: "Configuration file loaded successfully",
            type: "success",
          }),
        );
        setTimeout(() => dispatch(clearNotification()), 3000);
      })
      .catch((error) => {
        console.error("Error loading setup file:", error);
        dispatch(
          setNotification({
            message: "Failed to load configuration file",
            type: "error",
          }),
        );
      })
      .finally(() => {
        dispatch(uc2Slice.setIsLoadingFile(false));
      });
  };

  const handlePreviewConfig = () => {
    if (!newFileName) {
      dispatch(
        setNotification({
          message: "Please provide a filename",
          type: "warning",
        }),
      );
      return;
    }

    let finalJson = null;

    // Get JSON content based on editor type
    if (useAceEditor) {
      if (!editorJsonText.trim()) {
        dispatch(
          setNotification({
            message: "No JSON content to preview",
            type: "warning",
          }),
        );
        return;
      }

      const jsonValidation = validateJsonString(editorJsonText);
      if (!jsonValidation.isValid) {
        dispatch(
          setNotification({
            message: `Invalid JSON: ${jsonValidation.error}`,
            type: "error",
          }),
        );
        return;
      }
      finalJson = jsonValidation.parsed;
    } else {
      if (!editorJson) {
        dispatch(
          setNotification({
            message: "No JSON content to preview",
            type: "warning",
          }),
        );
        return;
      }
      finalJson = editorJson;
    }

    // Validate configuration
    const validation = validateConfiguration(finalJson);
    dispatch(uc2Slice.setValidationResult(validation));

    // Create preview
    const preview = createConfigurationPreview(finalJson);
    dispatch(uc2Slice.setConfigPreview(preview));

    // Show preview dialog
    dispatch(uc2Slice.setShowPreviewDialog(true));
  };

  const handleConfirmSave = () => {
    dispatch(uc2Slice.setShowPreviewDialog(false));
    handleSaveFile();
  };

  const handleSaveFile = () => {
    if (!newFileName) {
      dispatch(
        setNotification({
          message: "Please provide a filename",
          type: "warning",
        }),
      );
      return;
    }

    // Validate filename before processing
    const fileNameValidation = validateFileName(newFileName);
    if (!fileNameValidation.isValid) {
      dispatch(
        setNotification({
          message: fileNameValidation.error,
          type: "error",
        }),
      );
      return;
    }

    // Ensure filename ends with .json
    let finalFileName = newFileName.trim();
    if (!finalFileName.toLowerCase().endsWith(".json")) {
      finalFileName = finalFileName + ".json";
      // Update the state with the corrected filename
      dispatch(uc2Slice.setNewFileName(finalFileName));
    }

    let finalJson = null;
    if (useAceEditor) {
      if (!editorJsonText.trim()) {
        dispatch(
          setNotification({
            message: "No JSON content to save",
            type: "warning",
          }),
        );
        return;
      }

      const jsonValidation = validateJsonString(editorJsonText);
      if (!jsonValidation.isValid) {
        dispatch(
          setNotification({
            message: `Invalid JSON: ${jsonValidation.error}`,
            type: "error",
          }),
        );
        return;
      }
      finalJson = jsonValidation.parsed;
    } else {
      if (!editorJson) {
        dispatch(
          setNotification({
            message: "No JSON content to save",
            type: "warning",
          }),
        );
        return;
      }
      finalJson = editorJson;
    }

    dispatch(uc2Slice.setIsSavingFile(true));
    dispatch(
      setNotification({
        message: "Saving configuration file...",
        type: "info",
      }),
    );

    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/writeNewSetupFile?setupFileName=${encodeURIComponent(
      finalFileName,
    )}&setAsCurrentConfig=${setAsCurrentConfig}&restart=${restartAfterSave}&overwrite=${overwriteFile}`;

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(finalJson),
    })
      .then(async (response) => {
        if (!response.ok) {
          // Try to get error message from response
          let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
          try {
            const text = await response.text();
            // Check if it's JSON error
            try {
              const errorData = JSON.parse(text);
              errorMessage =
                errorData.message || errorData.error || errorMessage;
            } catch {
              // Not JSON, might be HTML error page
              if (text.includes("Internal Server Error")) {
                errorMessage =
                  "Backend Internal Server Error - Check backend logs for details";
              } else if (text.length < 200) {
                errorMessage = text;
              }
            }
          } catch {
            // Ignore text parsing errors
          }
          throw new Error(errorMessage);
        }
        return response.json();
      })
      .then((data) => {
        console.log("File saved:", data);
        dispatch(
          setNotification({
            message: "Configuration file saved successfully",
            type: "success",
          }),
        );

        // Refresh available setups
        fetchAvailableSetups();

        if (restartAfterSave) {
          dispatch(
            setNotification({
              message: "ImSwitch is restarting with new configuration...",
              type: "info",
            }),
          );
          monitorRestartStatus();
        } else {
          setTimeout(() => dispatch(clearNotification()), 3000);
        }
      })
      .catch((error) => {
        console.error("Error saving file:", error);
        dispatch(
          setNotification({
            message: `Failed to save configuration: ${error.message}`,
            type: "error",
          }),
        );
      })
      .finally(() => {
        dispatch(uc2Slice.setIsSavingFile(false));
      });
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1200, mx: "auto" }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          UC2 Configuration Manager
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure and manage your UC2 microscope system configurations
        </Typography>
      </Box>

      {/* Connection Status */}
      <Alert
        severity={uc2Connected ? "success" : "error"}
        sx={{ mb: 3 }}
        icon={uc2Connected ? <CheckCircle /> : <ErrorOutline />}
      >
        <Typography variant="body2">
          {uc2Connected ? (
            <>
              <strong>UC2 Board Connected:</strong> All controls are available.
            </>
          ) : (
            <>
              <strong>UC2 Board Disconnected:</strong> Please check your
              connection or try reconnecting.
            </>
          )}
        </Typography>
      </Alert>

      <Paper>
        <Tabs
          value={tabIndex}
          onChange={handleTabChange}
          aria-label="settings tabs"
        >
          <Tab label="Configuration Setup" />
          <Tab label="Advanced Editor" />
        </Tabs>

        <TabPanel value={tabIndex} index={0}>
          {/* Configuration Setup Card */}
          <Card>
            <CardContent>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}
              >
                <SettingsApplications color="primary" />
                <Typography variant="h6">Configuration Setup</Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={fetchAvailableSetups}
                  disabled={isRestarting}
                >
                  Refresh
                </Button>
              </Box>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Select and apply a predefined configuration for your UC2 system
              </Typography>

              {isRestarting && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Box sx={{ mb: 1 }}>
                    <LinearProgress />
                  </Box>
                  <Typography variant="body2">
                    {restartSoftware
                      ? "ImSwitch is restarting..."
                      : "Processing configuration..."}
                  </Typography>
                </Alert>
              )}

              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel id="setup-select-label">
                  Available Configurations
                </InputLabel>
                <Select
                  labelId="setup-select-label"
                  value={selectedSetup}
                  onChange={handleSetupChange}
                  disabled={isRestarting}
                >
                  {availableSetups.map((setup, index) => (
                    <MenuItem key={index} value={setup}>
                      {setup}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {currentActiveFilename && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Typography variant="body2">
                    Current configuration: {currentActiveFilename}
                  </Typography>
                </Alert>
              )}

              {/* Configuration Options */}
              <Box sx={{ mb: 3 }}>
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
                <Typography
                  variant="caption"
                  color="text.secondary"
                  display="block"
                >
                  Recommended: Automatically restart ImSwitch to apply the new
                  configuration
                </Typography>
              </Box>

              <Button
                variant="contained"
                onClick={handleSetSetup}
                disabled={!selectedSetup || isRestarting}
                startIcon={
                  isRestarting ? (
                    <CircularProgress size={20} />
                  ) : (
                    <SettingsApplications />
                  )
                }
                size="large"
                fullWidth
              >
                {isRestarting ? "Processing..." : "Apply Configuration"}
              </Button>

              {/* Status Information */}
              {selectedSetup && (
                <Box
                  sx={{
                    mt: 3,
                    p: 2,
                    bgcolor: "background.default",
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="subtitle2" gutterBottom>
                    Selected Configuration:
                  </Typography>
                  <Chip
                    label={selectedSetup}
                    color="primary"
                    size="small"
                    variant="outlined"
                  />
                </Box>
              )}

              {/* Confirmation Dialog */}
              <Dialog open={isDialogOpen} onClose={handleDialogClose}>
                <DialogTitle>Configuration Update</DialogTitle>
                <DialogContent>
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Configuration update completed successfully!
                  </Alert>
                  <Typography>
                    The setup has been updated.
                    {restartSoftware &&
                      " The system is restarting and will be back online shortly."}
                  </Typography>
                </DialogContent>
                <DialogActions>
                  <Button
                    onClick={handleDialogClose}
                    color="primary"
                    variant="contained"
                  >
                    OK
                  </Button>
                </DialogActions>
              </Dialog>
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabIndex} index={1}>
          {/* Configuration Wizard Recommendation */}
          <Card
            sx={{
              mb: 3,
              bgcolor: "primary.light",
              color: "primary.contrastText",
            }}
          >
            <CardContent>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}
              >
                <WizardIcon />
                <Typography variant="h6">
                  Recommended: Configuration Wizard
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mb: 3 }}>
                Get guided through the configuration process step-by-step with
                built-in validation, preview, and easy save options. Perfect for
                beginners and complex configurations.
              </Typography>
              <Button
                variant="contained"
                size="large"
                startIcon={<WizardIcon />}
                onClick={() => setShowConfigWizard(true)}
                sx={{
                  bgcolor: "background.paper",
                  color: "primary.main",
                  "&:hover": { bgcolor: "grey.100" },
                }}
              >
                Launch Configuration Wizard
              </Button>
            </CardContent>
          </Card>

          {/* Advanced Editor Card */}
          <Card>
            <CardContent>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 2, mb: 3 }}
              >
                <Build color="secondary" />
                <Typography variant="h6">
                  Advanced Configuration Editor
                </Typography>
              </Box>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                For advanced users: directly edit JSON configuration files with
                syntax validation and preview
              </Typography>

              {(isLoadingFile || isSavingFile) && (
                <Alert severity="info" sx={{ mb: 3 }}>
                  <Box sx={{ mb: 1 }}>
                    <LinearProgress />
                  </Box>
                  <Typography variant="body2">
                    {isLoadingFile
                      ? "Loading configuration file..."
                      : "Saving configuration file..."}
                  </Typography>
                </Alert>
              )}

              {/* File Selection Section */}
              <Box sx={{ mb: 3 }}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="editor-select-label">
                    Select Configuration File to Load
                  </InputLabel>
                  <Select
                    labelId="editor-select-label"
                    value={selectedFileForEdit}
                    onChange={(e) =>
                      dispatch(uc2Slice.setSelectedFileForEdit(e.target.value))
                    }
                    disabled={isLoadingFile || isSavingFile}
                  >
                    {availableSetups.map((setup, index) => (
                      <MenuItem key={index} value={setup}>
                        {setup}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
                  <Button
                    variant="contained"
                    onClick={handleLoadSetupFile}
                    disabled={
                      !selectedFileForEdit || isLoadingFile || isSavingFile
                    }
                    startIcon={
                      isLoadingFile ? <CircularProgress size={20} /> : null
                    }
                  >
                    {isLoadingFile ? "Loading..." : "Load File"}
                  </Button>
                  <Button
                    variant="contained"
                    onClick={handleNewConfig}
                    disabled={isLoadingFile || isSavingFile}
                  >
                    New File
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<RefreshIcon />}
                    onClick={fetchAvailableSetups}
                    disabled={isLoadingFile || isSavingFile}
                    size="small"
                  >
                    Refresh List
                  </Button>
                </Box>
              </Box>

              {/* JSON Editor Section */}
              <Box sx={{ mb: 3 }}>
                {useAceEditor ? (
                  <>
                    <Typography variant="subtitle1" gutterBottom>
                      JSON Configuration (AceEditor for new files):
                    </Typography>
                    <AceEditor
                      mode="json"
                      setOptions={{ useWorker: false }}
                      theme="github"
                      onChange={(value) =>
                        dispatch(uc2Slice.setEditorJsonText(value))
                      }
                      value={editorJsonText}
                      name="jsonEditor"
                      editorProps={{ $blockScrolling: true }}
                      width="100%"
                      height="400px"
                      readOnly={isLoadingFile || isSavingFile}
                    />
                  </>
                ) : editorJson ? (
                  <>
                    <Typography variant="subtitle1" gutterBottom>
                      JSON Configuration (Interactive editor for loaded files):
                    </Typography>
                    <Box
                      sx={{
                        height: "400px",
                        overflow: "auto",
                        border: "1px solid",
                        borderColor: "grey.300",
                        borderRadius: 1,
                      }}
                    >
                      <JsonEditor
                        data={editorJson}
                        setData={(data) =>
                          dispatch(uc2Slice.setEditorJson(data))
                        }
                        readOnly={isLoadingFile || isSavingFile}
                      />
                    </Box>
                  </>
                ) : (
                  <Alert severity="info">
                    <Typography variant="body2">
                      No configuration file loaded. Select a file to load or
                      create a new one.
                    </Typography>
                  </Alert>
                )}
              </Box>

              {/* Save Section */}
              <Box>
                <TextField
                  label="Save As Filename"
                  fullWidth
                  value={newFileName}
                  onChange={(e) =>
                    dispatch(uc2Slice.setNewFileName(e.target.value))
                  }
                  sx={{ mb: 3 }}
                  disabled={isLoadingFile || isSavingFile}
                  placeholder="my_configuration.json"
                  helperText=".json extension will be added automatically if missing"
                />

                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={12} sm={4}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={setAsCurrentConfig}
                          onChange={(e) =>
                            dispatch(
                              uc2Slice.setSetAsCurrentConfig(e.target.checked),
                            )
                          }
                          disabled={isLoadingFile || isSavingFile}
                        />
                      }
                      label="Set as Current Config"
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={restartAfterSave}
                          onChange={(e) =>
                            dispatch(
                              uc2Slice.setRestartAfterSave(e.target.checked),
                            )
                          }
                          disabled={isLoadingFile || isSavingFile}
                        />
                      }
                      label="Restart After Save"
                    />
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={overwriteFile}
                          onChange={(e) =>
                            dispatch(
                              uc2Slice.setOverwriteFile(e.target.checked),
                            )
                          }
                          disabled={isLoadingFile || isSavingFile}
                        />
                      }
                      label="Overwrite if exists"
                    />
                  </Grid>
                </Grid>

                <Box sx={{ display: "flex", gap: 1 }}>
                  <Button
                    variant="outlined"
                    onClick={handlePreviewConfig}
                    disabled={
                      !newFileName ||
                      isLoadingFile ||
                      isSavingFile ||
                      (!editorJson && !editorJsonText.trim())
                    }
                    startIcon={<PreviewIcon />}
                  >
                    Preview & Validate
                  </Button>
                  <Button
                    variant="contained"
                    onClick={handleSaveFile}
                    disabled={
                      !newFileName ||
                      isLoadingFile ||
                      isSavingFile ||
                      (!editorJson && !editorJsonText.trim())
                    }
                    startIcon={
                      isSavingFile ? (
                        <CircularProgress size={20} />
                      ) : (
                        <SaveIcon />
                      )
                    }
                  >
                    {isSavingFile ? "Saving..." : "Save Configuration"}
                  </Button>
                </Box>
              </Box>

              {/* Configuration Preview Dialog */}
              <ConfigurationPreviewDialog
                open={showPreviewDialog}
                onClose={() => dispatch(uc2Slice.setShowPreviewDialog(false))}
                onConfirmSave={handleConfirmSave}
                validationResult={validationResult}
                configPreview={configPreview}
                filename={newFileName}
                isSaving={isSavingFile}
              />
            </CardContent>
          </Card>
        </TabPanel>
      </Paper>

      {/* Configuration Wizard */}
      <ConfigurationWizard
        open={showConfigWizard}
        onClose={() => setShowConfigWizard(false)}
        hostIP={hostIP}
        hostPort={hostPort}
      />
    </Box>
  );
};

export default UC2ConfigurationController;
