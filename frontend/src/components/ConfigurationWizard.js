import React, { useState, useCallback, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  Box,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import * as uc2Slice from "../state/slices/UC2Slice";
import { validateJsonString } from "../utils/configValidation";

// Import wizard steps
import ConfigWizardStep1 from "./config-wizard-steps/ConfigWizardStep1";
import ConfigWizardStep2 from "./config-wizard-steps/ConfigWizardStep2";
import ConfigWizardStep3 from "./config-wizard-steps/ConfigWizardStep3";
import ConfigWizardStep4 from "./config-wizard-steps/ConfigWizardStep4";
import ConfigWizardStep5 from "./config-wizard-steps/ConfigWizardStep5";
import ConfigWizardStep6 from "./config-wizard-steps/ConfigWizardStep6";

const steps = [
  "Choose Source",
  "Load Configuration", 
  "Edit & Validate",
  "Save Options",
  "Execute Save",
  "Complete"
];

const ConfigurationWizard = ({ open, onClose, hostIP, hostPort }) => {
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));
  const dispatch = useDispatch();
  
  // Redux state
  const uc2State = useSelector(uc2Slice.getUc2State);
  const availableSetups = uc2State.availableSetups;
  
  // Wizard state
  const [activeStep, setActiveStep] = useState(0);
  const [selectedSource, setSelectedSource] = useState('');
  const [selectedFile, setSelectedFile] = useState('');
  const [currentActiveFilename, setCurrentActiveFilename] = useState('');
  const [loadedConfig, setLoadedConfig] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  
  // Editor state
  const [editorJson, setEditorJson] = useState(null);
  const [editorJsonText, setEditorJsonText] = useState('');
  const [useAceEditor, setUseAceEditor] = useState(false);
  
  // Save options state
  const [filename, setFilename] = useState('');
  const [setAsCurrentConfig, setSetAsCurrentConfig] = useState(true);
  const [restartAfterSave, setRestartAfterSave] = useState(false);
  const [overwriteFile, setOverwriteFile] = useState(false);
  
  // Save execution state
  const [isSaving, setIsSaving] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [restartProgress, setRestartProgress] = useState({ current: 0, max: 30 });

  // Fetch available setups when wizard opens
  useEffect(() => {
    if (open && availableSetups.length === 0) {
      fetchAvailableSetups();
    }
  }, [open]);

  // Auto-populate filename when entering Step 4 with current configuration
  useEffect(() => {
    if (activeStep === 3 && selectedSource === 'current' && currentActiveFilename && !filename) {
      // Extract filename without extension if needed, or use as-is
      const baseFilename = typeof currentActiveFilename === 'string' 
        ? currentActiveFilename
        : JSON.stringify(currentActiveFilename);
      setFilename(baseFilename);
    }
  }, [activeStep, selectedSource, currentActiveFilename, filename]);

  // Fetch current active setup filename when "current" is selected
  useEffect(() => {
    if (selectedSource === 'current' && !currentActiveFilename) {
      fetchCurrentActiveFilename();
    }
  }, [selectedSource]);

  const fetchCurrentActiveFilename = useCallback(() => {
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/getCurrentSetupFilename`;
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
        
        setCurrentActiveFilename(filename || 'current_config.json');
      })
      .catch((error) => {
        console.error("Error fetching current setup filename:", error);
        setCurrentActiveFilename('current_config.json'); // fallback
      });
  }, [hostIP, hostPort]);

  const fetchAvailableSetups = useCallback(() => {
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/returnAvailableSetups`;
    fetch(url)
      .then((response) => response.json())
      .then((data) => {
        dispatch(uc2Slice.setAvailableSetups(data.available_setups || []));
      })
      .catch((error) => {
        console.error("Error fetching setups:", error);
      });
  }, [hostIP, hostPort, dispatch]);

  // Step navigation
  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleStepClick = (step) => {
    // Allow clicking on completed steps or the next step
    if (step <= activeStep + 1) {
      setActiveStep(step);
    }
  };

  const resetWizard = () => {
    setActiveStep(0);
    setSelectedSource('');
    setSelectedFile('');
    setCurrentActiveFilename('');
    setLoadedConfig(null);
    setIsLoading(false);
    setLoadError('');
    setEditorJson(null);
    setEditorJsonText('');
    setUseAceEditor(false);
    setFilename('');
    setSetAsCurrentConfig(true);
    setRestartAfterSave(false);
    setOverwriteFile(false);
    setIsSaving(false);
    setIsRestarting(false);
    setSaveError('');
    setSaveSuccess(false);
    setRestartProgress({ current: 0, max: 30 });
  };

  const handleClose = () => {
    resetWizard();
    onClose();
  };

  // Configuration loading functions
  const handleLoadCurrent = useCallback(() => {
    setIsLoading(true);
    setLoadError('');
    
    // Use the readSetupFile API without setupFileName parameter to get current config
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/readSetupFile`;
    
    fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setLoadedConfig(data);
        setEditorJson(data);
        setEditorJsonText(JSON.stringify(data, null, 2));
        setUseAceEditor(false);
        setIsLoading(false);
      })
      .catch((error) => {
        setLoadError(error.message || 'Failed to load current configuration');
        setIsLoading(false);
      });
  }, [hostIP, hostPort]);

  const handleLoadFile = useCallback((fileName) => {
    setIsLoading(true);
    setLoadError('');
    
    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/readSetupFile?setupFileName=${encodeURIComponent(fileName)}`;
    
    fetch(url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        setLoadedConfig(data);
        setEditorJson(data);
        setEditorJsonText(JSON.stringify(data, null, 2));
        setUseAceEditor(false);
        setIsLoading(false);
      })
      .catch((error) => {
        setLoadError(error.message || 'Failed to load configuration file');
        setIsLoading(false);
      });
  }, [hostIP, hostPort]);

  const handleCreateNew = useCallback(() => {
    setIsLoading(true);
    
    setTimeout(() => {
      const newConfig = {
        "detectors": {},
        "lasers": {},
        "LEDs": {},
        "LEDMatrixs": {},
        "positioners": {},
        "rs232devices": {},
        "slm": null,
        "sim": null,
        "dpc": null,
        "objective": null,
        "nidaq": {
          "timerCounterChannel": null,
          "startTrigger": false
        },
        "roiscan": null,
        "lightsheet": null,
        "webrtc": null,
        "hypha": null,
        "mockxx": null,
        "jetsonnano": null,
        "Stresstest": {},
        "HistoScan": null,
        "Workflow": null,
        "FlowStop": null,
        "Lepmon": null,
        "Flatfield": null,
        "PixelCalibration": {},
        "uc2Config": null,
        "ism": null,
        "focusLock": null,
        "fovLock": null,
        "autofocus": null,
        "scan": null,
        "etSTED": null,
        "rotators": null,
        "microscopeStand": null,
        "pulseStreamer": {
          "ipAddress": null
        },
        "pyroServerInfo": {
          "name": "ImSwitchServer",
          "host": "0.0.0.0",
          "port": 54333,
          "active": false
        },
        "rois": {
          "Full chip": {
            "x": 600,
            "y": 600,
            "w": 1200,
            "h": 1200
          }
        },
        "ledPresets": {},
        "defaultLEDPresetForScan": null,
        "laserPresets": {},
        "stageOffsets": {},
        "defaultLaserPresetForScan": null,
        "availableWidgets": [],
        "nonAvailableWidgets": []
      };
      
      setLoadedConfig(newConfig);
      setEditorJson(newConfig);
      setEditorJsonText(JSON.stringify(newConfig, null, 2));
      setUseAceEditor(false);
      setIsLoading(false);
    }, 500);
  }, []);

  // Save execution
  const handleExecuteSave = useCallback(() => {
    setIsSaving(true);
    setSaveError('');
    setSaveSuccess(false);
    
    // Get the final configuration
    let finalJson = null;
    if (useAceEditor) {
      const jsonValidation = validateJsonString(editorJsonText);
      if (!jsonValidation.isValid) {
        setSaveError(`Invalid JSON: ${jsonValidation.error}`);
        setIsSaving(false);
        return;
      }
      finalJson = jsonValidation.parsed;
    } else {
      finalJson = editorJson || loadedConfig;
    }

    if (!finalJson) {
      setSaveError('No configuration to save');
      setIsSaving(false);
      return;
    }

    const url = `${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/writeNewSetupFile?setupFileName=${encodeURIComponent(
      filename
    )}&setAsCurrentConfig=${setAsCurrentConfig}&restart=${restartAfterSave}&overwrite=${overwriteFile}`;

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(finalJson),
    })
      .then((response) => response.json())
      .then((data) => {
        setSaveSuccess(true);
        setIsSaving(false);
        
        if (restartAfterSave) {
          setIsRestarting(true);
          monitorRestart();
        }
        
        // Refresh available setups
        fetchAvailableSetups();
      })
      .catch((error) => {
        setSaveError(error.message || 'Failed to save configuration');
        setIsSaving(false);
      });
  }, [filename, setAsCurrentConfig, restartAfterSave, overwriteFile, useAceEditor, editorJsonText, editorJson, loadedConfig, hostIP, hostPort, fetchAvailableSetups]);

  const monitorRestart = useCallback(() => {
    let retryCount = 0;
    const maxRetries = 30;
    
    const checkStatus = () => {
      fetch(`${hostIP}:${hostPort}/imswitch/api/UC2ConfigController/is_connected`)
        .then((res) => res.json())
        .then((data) => {
          if (data === true) {
            setIsRestarting(false);
            setRestartProgress({ current: maxRetries, max: maxRetries });
          } else {
            throw new Error("Not connected yet");
          }
        })
        .catch(() => {
          retryCount++;
          setRestartProgress({ current: retryCount, max: maxRetries });
          
          if (retryCount < maxRetries) {
            setTimeout(checkStatus, 10000);
          } else {
            setIsRestarting(false);
            setSaveError('System restart took longer than expected. Please check manually.');
          }
        });
    };
    
    setTimeout(checkStatus, 5000);
  }, [hostIP, hostPort]);

  // Validation functions for step transitions
  const canProceedFromStep = (step) => {
    switch (step) {
      case 0: return selectedSource !== '';
      case 1: return loadedConfig !== null || (useAceEditor && editorJsonText.trim() !== '');
      case 2: return loadedConfig !== null || (useAceEditor && editorJsonText.trim() !== '');
      case 3: return filename && (!availableSetups.includes(filename) || overwriteFile);
      case 4: return saveSuccess;
      default: return true;
    }
  };

  const getStepContent = (step) => {
    const commonProps = {
      onNext: handleNext,
      onBack: handleBack,
      activeStep,
      totalSteps: steps.length,
    };

    switch (step) {
      case 0:
        return (
          <ConfigWizardStep1
            {...commonProps}
            selectedSource={selectedSource}
            onSourceChange={setSelectedSource}
            currentActiveFilename={currentActiveFilename}
          />
        );
      case 1:
        return (
          <ConfigWizardStep2
            {...commonProps}
            selectedSource={selectedSource}
            availableSetups={availableSetups}
            selectedFile={selectedFile}
            onFileChange={setSelectedFile}
            loadedConfig={loadedConfig}
            isLoading={isLoading}
            loadError={loadError}
            onLoadCurrent={handleLoadCurrent}
            onLoadFile={handleLoadFile}
            onCreateNew={handleCreateNew}
            onRefreshFiles={fetchAvailableSetups}
            currentActiveFilename={currentActiveFilename}
          />
        );
      case 2:
        return (
          <ConfigWizardStep3
            {...commonProps}
            loadedConfig={loadedConfig}
            editorJson={editorJson}
            editorJsonText={editorJsonText}
            useAceEditor={useAceEditor}
            onConfigChange={setEditorJson}
            onJsonTextChange={setEditorJsonText}
            onEditorTypeChange={setUseAceEditor}
          />
        );
      case 3:
        return (
          <ConfigWizardStep4
            {...commonProps}
            filename={filename}
            setAsCurrentConfig={setAsCurrentConfig}
            restartAfterSave={restartAfterSave}
            overwriteFile={overwriteFile}
            onFilenameChange={setFilename}
            onSetAsCurrentConfigChange={setSetAsCurrentConfig}
            onRestartAfterSaveChange={setRestartAfterSave}
            onOverwriteFileChange={setOverwriteFile}
            availableSetups={availableSetups}
          />
        );
      case 4:
        return (
          <ConfigWizardStep5
            {...commonProps}
            filename={filename}
            setAsCurrentConfig={setAsCurrentConfig}
            restartAfterSave={restartAfterSave}
            overwriteFile={overwriteFile}
            isSaving={isSaving}
            isRestarting={isRestarting}
            saveError={saveError}
            saveSuccess={saveSuccess}
            restartProgress={restartProgress}
            onExecuteSave={handleExecuteSave}
          />
        );
      case 5:
        return (
          <ConfigWizardStep6
            {...commonProps}
            filename={filename}
            setAsCurrentConfig={setAsCurrentConfig}
            restartAfterSave={restartAfterSave}
            saveSuccess={saveSuccess}
            onStartNewWizard={resetWizard}
            onComplete={handleClose}
          />
        );
      default:
        return <ConfigWizardStep1 {...commonProps} />;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="xl"
      fullWidth
      fullScreen={fullScreen}
      PaperProps={{
        style: {
          minHeight: '80vh',
        },
      }}
    >
      <DialogTitle>
        Configuration Wizard
      </DialogTitle>
      
      <DialogContent>
        <Box sx={{ width: '100%', mb: 2 }}>
          <Stepper 
            activeStep={activeStep} 
            alternativeLabel={!fullScreen}
            orientation={fullScreen ? "vertical" : "horizontal"}
          >
            {steps.map((label, index) => {
              const stepProps = {};
              const labelProps = {};
              
              // Mark completed steps
              if (index < activeStep) {
                stepProps.completed = true;
              }
              
              return (
                <Step 
                  key={label} 
                  {...stepProps}
                >
                  <StepLabel 
                    {...labelProps}
                    sx={{ 
                      cursor: index <= activeStep + 1 ? 'pointer' : 'default',
                      '& .MuiStepLabel-label': {
                        cursor: index <= activeStep + 1 ? 'pointer' : 'default'
                      }
                    }}
                    onClick={() => handleStepClick(index)}
                  >
                    {label}
                  </StepLabel>
                </Step>
              );
            })}
          </Stepper>
        </Box>
        
        <Box sx={{ mt: 2, minHeight: '60vh' }}>
          {getStepContent(activeStep)}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose} color="secondary">
          Cancel
        </Button>
        <Box sx={{ flex: '1 1 auto' }} />
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
          sx={{ mr: 1 }}
        >
          Back
        </Button>
        {activeStep === steps.length - 1 ? (
          <Button onClick={handleClose} variant="contained" color="primary">
            Finish
          </Button>
        ) : (
          <Button 
            onClick={handleNext} 
            variant="contained" 
            color="primary"
            disabled={!canProceedFromStep(activeStep)}
          >
            Next
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default ConfigurationWizard;
