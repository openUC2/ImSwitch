import React, { useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Stepper,
  Step,
  StepLabel,
  Box,
  useTheme,
  useMediaQuery,
  IconButton,
} from "@mui/material";
import { Close } from "@mui/icons-material";
import * as stageCenterCalibrationSlice from "../state/slices/StageCenterCalibrationSlice";

// Import wizard steps
import StageCenterStep1 from "./wizard-steps/StageCenterStep1";
import StageCenterStep2 from "./wizard-steps/StageCenterStep2";
import StageCenterStep3 from "./wizard-steps/StageCenterStep3";
import StageCenterStep4 from "./wizard-steps/StageCenterStep4";
import StageCenterStep5 from "./wizard-steps/StageCenterStep5";
import StageCenterStep6 from "./wizard-steps/StageCenterStep6";

const steps = [
  "Setup & Overview",
  "Manual Position Entry",
  "Stage Map Visualization",
  "Automatic Detection",
  "Review Results",
  "Complete"
];

const StageCenterCalibrationWizard = () => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('md'));
  
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  const { isWizardOpen, activeStep } = stageCenterState;

  // Auto-open wizard when component mounts
  useEffect(() => {
    dispatch(stageCenterCalibrationSlice.setWizardOpen(true));
    // Clean up: close wizard when component unmounts
    return () => {
      dispatch(stageCenterCalibrationSlice.setWizardOpen(false));
    };
  }, [dispatch]);

  const handleNext = () => {
    dispatch(stageCenterCalibrationSlice.nextStep());
  };

  const handleBack = () => {
    dispatch(stageCenterCalibrationSlice.previousStep());
  };

  const handleStepClick = (step) => {
    dispatch(stageCenterCalibrationSlice.setActiveStep(step));
  };

  const handleClose = () => {
    dispatch(stageCenterCalibrationSlice.setWizardOpen(false));
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
        return <StageCenterStep1 {...commonProps} />;
      case 1:
        return <StageCenterStep2 {...commonProps} />;
      case 2:
        return <StageCenterStep3 {...commonProps} />;
      case 3:
        return <StageCenterStep4 {...commonProps} />;
      case 4:
        return <StageCenterStep5 {...commonProps} />;
      case 5:
        return <StageCenterStep6 {...commonProps} onComplete={handleClose} />;
      default:
        return <StageCenterStep1 {...commonProps} />;
    }
  };

  return (
    <Dialog
      open={isWizardOpen}
      onClose={handleClose}
      maxWidth="xl"
      fullWidth
      fullScreen={fullScreen}
      PaperProps={{
        style: {
          minHeight: '80vh',
          background: theme.palette.background.default,
          color: theme.palette.text.primary,
        },
      }}
    >
      <DialogTitle
        sx={{
          background: theme.palette.background.paper,
          color: theme.palette.text.primary,
          fontWeight: 'bold',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        Stage Center Calibration Wizard
        <IconButton onClick={handleClose} color="inherit">
          <Close />
        </IconButton>
      </DialogTitle>
      
      <DialogContent
        sx={{
          background: theme.palette.background.default,
          color: theme.palette.text.primary,
        }}
      >
        <Box sx={{ width: '100%', mb: 2 }}>
          <Stepper 
            activeStep={activeStep} 
            alternativeLabel={!fullScreen}
            orientation={fullScreen ? "vertical" : "horizontal"}
            sx={{
              '& .MuiStepLabel-root .Mui-completed': {
                color: '#4caf50',
              },
              '& .MuiStepLabel-root .Mui-active': {
                color: '#1976d2',
              },
            }}
          >
            {steps.map((label, index) => {
              const stepProps = {};
              const labelProps = {};
              
              return (
                <Step 
                  key={label} 
                  {...stepProps}
                  sx={{ cursor: 'pointer' }}
                  onClick={() => handleStepClick(index)}
                >
                  <StepLabel 
                    {...labelProps}
                    sx={{
                      '& .MuiStepLabel-label': {
                        fontSize: fullScreen ? '0.875rem' : '1rem',
                        fontWeight: activeStep === index ? 'bold' : 'normal',
                      }
                    }}
                  >
                    {label}
                  </StepLabel>
                </Step>
              );
            })}
          </Stepper>
        </Box>
        
        <Box sx={{ mt: 3 }}>
          {getStepContent(activeStep)}
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default StageCenterCalibrationWizard;