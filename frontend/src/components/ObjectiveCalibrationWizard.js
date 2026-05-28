import React, { useState } from "react";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
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
import WizardStep1 from "./wizard-steps/WizardStep1";
import WizardStep2 from "./wizard-steps/WizardStep2";
import WizardStep3 from "./wizard-steps/WizardStep3";
import WizardStep4 from "./wizard-steps/WizardStep4";
import WizardStep5 from "./wizard-steps/WizardStep5";
import WizardStep6 from "./wizard-steps/WizardStep6";

const ObjectiveCalibrationWizard = ({ open, onClose }) => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;

  // Check whether slot 1 is configured to filter out slot-1 steps
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);
  const slot1Configured = objectiveState.slotConfigured?.[1] ?? true;

  // Build the step list dynamically so we skip slot-1 steps in single-objective mode
  const stepDefs = [
    { label: "Setup Instructions",    Component: WizardStep1 },
    { label: "Calibrate Slot 1 (X0)", Component: WizardStep2 },
    ...(slot1Configured ? [{ label: "Calibrate Slot 2 (X1)", Component: WizardStep3 }] : []),
    { label: "Calibrate Focus Z0",    Component: WizardStep4 },
    ...(slot1Configured ? [{ label: "Calibrate Focus Z1",    Component: WizardStep5 }] : []),
    { label: "Complete",              Component: WizardStep6 },
  ];
  const steps = stepDefs.map((d) => d.label);

  const [activeStep, setActiveStep] = useState(0);
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));

  const handleNext = () => {
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  const handleStepClick = (step) => {
    setActiveStep(step);
  };

  const handleClose = () => {
    setActiveStep(0); // Reset to first step when closing
    onClose();
  };

  const getStepContent = (step) => {
    const def = stepDefs[step];
    if (!def) return null;
    const { Component } = def;
    const commonProps = {
      hostIP,
      hostPort,
      onNext: handleNext,
      onBack: handleBack,
      activeStep,
      totalSteps: steps.length,
    };
    const extra = step === stepDefs.length - 1 ? { onComplete: handleClose } : {};
    return <Component {...commonProps} {...extra} />;
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
        Objective Calibration Wizard
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
              
              return (
                <Step 
                  key={label} 
                  completed={index < activeStep}
                  {...stepProps}
                >
                  <StepLabel 
                    {...labelProps}
                    sx={{ 
                      cursor: 'pointer',
                      '& .MuiStepLabel-label': {
                        cursor: 'pointer'
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
          <Button onClick={handleNext} variant="contained" color="primary">
            Next
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default ObjectiveCalibrationWizard;