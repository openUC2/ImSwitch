import React, { useEffect, useState } from "react";
import {
  Typography,
  Box,
  Paper,
  Button,
  Alert,
  LinearProgress,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
} from "@mui/material";
import {
  Save as SaveIcon,
  Settings as SettingsIcon,
  RestartAlt as RestartIcon,
  Check as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  HourglassFull as HourglassIcon,
} from "@mui/icons-material";

const ConfigWizardStep5 = ({ 
  filename,
  setAsCurrentConfig,
  restartAfterSave,
  overwriteFile,
  isSaving,
  isRestarting,
  saveError,
  saveSuccess,
  restartProgress,
  onExecuteSave,
  onNext, 
  onBack, 
  activeStep, 
  totalSteps 
}) => {
  const [hasExecuted, setHasExecuted] = useState(false);

  const getStatusInfo = () => {
    if (saveError) {
      return {
        type: 'error',
        title: 'Save Operation Failed',
        message: saveError,
        icon: <ErrorIcon color="error" />
      };
    }
    
    if (isRestarting) {
      return {
        type: 'info',
        title: 'System Restarting',
        message: `Waiting for system to restart... (${restartProgress?.current || 0}/${restartProgress?.max || 30})`,
        icon: <RestartIcon color="info" />
      };
    }
    
    if (isSaving) {
      return {
        type: 'info',
        title: 'Saving Configuration',
        message: 'Please wait while the configuration is being saved...',
        icon: <SaveIcon color="info" />
      };
    }
    
    if (saveSuccess && !restartAfterSave) {
      return {
        type: 'success',
        title: 'Configuration Saved Successfully',
        message: 'Your configuration has been saved and is ready to use.',
        icon: <CheckIcon color="success" />
      };
    }
    
    if (saveSuccess && restartAfterSave && !isRestarting) {
      return {
        type: 'success',
        title: 'Configuration Applied Successfully',
        message: 'Your configuration has been saved and the system has restarted successfully.',
        icon: <CheckIcon color="success" />
      };
    }
    
    return {
      type: 'info',
      title: 'Ready to Execute',
      message: 'Review your settings and click "Execute Save" to apply the configuration.',
      icon: <HourglassIcon color="info" />
    };
  };

  const status = getStatusInfo();
  const isComplete = saveSuccess && (!restartAfterSave || !isRestarting);
  const isInProgress = isSaving || isRestarting;
  const canExecute = !hasExecuted && !isInProgress && !saveError;

  const handleExecute = () => {
    setHasExecuted(true);
    onExecuteSave();
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Execute Save Operation
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Review your configuration settings below and execute the save operation when ready.
      </Alert>

      {/* Status Section */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {status.icon}
          <Typography variant="h6" sx={{ ml: 1 }}>
            {status.title}
          </Typography>
        </Box>
        
        <Typography variant="body2" color="text.secondary" paragraph>
          {status.message}
        </Typography>
        
        {/* Progress Indicators */}
        {isInProgress && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress sx={{ mb: 1 }} />
            {isRestarting && restartProgress && (
              <Typography variant="caption" color="text.secondary">
                Restart progress: {restartProgress.current} of {restartProgress.max} attempts
              </Typography>
            )}
          </Box>
        )}
        
        {/* Status Chips */}
        <Box sx={{ display: 'flex', gap: 1, mt: 2, flexWrap: 'wrap' }}>
          <Chip 
            label={isSaving ? "Saving..." : saveSuccess ? "Saved" : "Ready"} 
            color={saveSuccess ? "success" : isSaving ? "info" : "default"}
            icon={isSaving ? <CircularProgress size={16} /> : saveSuccess ? <CheckIcon /> : undefined}
          />
          
          {restartAfterSave && (
            <Chip 
              label={isRestarting ? "Restarting..." : (saveSuccess && !isRestarting) ? "Restarted" : "Will Restart"} 
              color={saveSuccess && !isRestarting ? "success" : isRestarting ? "warning" : "default"}
              icon={isRestarting ? <CircularProgress size={16} /> : (saveSuccess && !isRestarting) ? <CheckIcon /> : <RestartIcon />}
            />
          )}
          
          {setAsCurrentConfig && (
            <Chip 
              label={saveSuccess ? "Set as Active" : "Will Set as Active"} 
              color={saveSuccess ? "success" : "default"}
              icon={saveSuccess ? <CheckIcon /> : <SettingsIcon />}
            />
          )}
        </Box>
      </Paper>

      {/* Configuration Summary */}
      <Paper elevation={1} sx={{ p: 3, mb: 3, backgroundColor: 'secondary' }}>
        <Typography variant="h6" gutterBottom>
          Configuration Summary
        </Typography>
        
        <List dense>
          <ListItem>
            <ListItemIcon>
              <SaveIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary={`Filename: ${filename}`}
              secondary={overwriteFile ? "Will overwrite existing file" : "Will create new file"}
            />
          </ListItem>
          
          <Divider variant="inset" component="li" />
          
          <ListItem>
            <ListItemIcon>
              <SettingsIcon color={setAsCurrentConfig ? "primary" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary={setAsCurrentConfig ? "Set as Active Configuration" : "Save Only"}
              secondary={setAsCurrentConfig ? "This will become the system's active configuration" : "Configuration will be saved but not activated"}
            />
          </ListItem>
          
          <Divider variant="inset" component="li" />
          
          <ListItem>
            <ListItemIcon>
              <RestartIcon color={restartAfterSave ? "warning" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary={restartAfterSave ? "System Restart Enabled" : "No System Restart"}
              secondary={restartAfterSave ? "System will restart automatically after saving" : "System will continue with current configuration"}
            />
          </ListItem>
        </List>
      </Paper>

      {/* Action Buttons */}
      {canExecute && (
        <Paper elevation={1} sx={{ p: 3, mb: 3, backgroundColor: 'success.light' }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <CheckIcon sx={{ mr: 1 }} />
            Ready to Execute
          </Typography>
          <Typography variant="body2" paragraph>
            All settings have been configured. Click the button below to save your configuration.
          </Typography>
          <Button
            variant="contained"
            size="large"
            startIcon={<SaveIcon />}
            onClick={handleExecute}
            sx={{ mt: 1 }}
          >
            Execute Save Operation
          </Button>
        </Paper>
      )}

      {/* Error Display */}
      {saveError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Save operation failed:</strong><br />
            {saveError}
          </Typography>
        </Alert>
      )}

      {/* Restart Warning */}
      {restartAfterSave && hasExecuted && !saveError && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>System Restart in Progress:</strong><br />
            The system is restarting with your new configuration. This process may take several minutes.
            Please do not close this wizard until the restart is complete.
          </Typography>
        </Alert>
      )}

      {/* Success Message */}
      {isComplete && (
        <Alert severity="success" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>Configuration Applied Successfully!</strong><br />
            Your configuration has been saved and is now active. You can proceed to the final step.
          </Typography>
        </Alert>
      )}
    </Box>
  );
};

export default ConfigWizardStep5;