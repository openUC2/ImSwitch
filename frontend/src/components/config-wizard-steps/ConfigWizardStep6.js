import React from "react";
import {
  Typography,
  Box,
  Paper,
  Button,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Chip,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Settings as SettingsIcon,
  RestartAlt as RestartIcon,
  Refresh as RefreshIcon,
  PlayArrow as PlayArrowIcon,
  Info as InfoIcon,
} from "@mui/icons-material";

const ConfigWizardStep6 = ({ 
  filename,
  setAsCurrentConfig,
  restartAfterSave,
  saveSuccess,
  onStartNewWizard,
  onComplete,
  onNext, 
  onBack, 
  activeStep, 
  totalSteps 
}) => {
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Configuration Wizard Complete
      </Typography>
      
      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
          <CheckCircleIcon sx={{ mr: 1 }} />
          Congratulations! Your configuration has been successfully applied.
        </Typography>
      </Alert>

      {/* Success Summary */}
      <Paper elevation={2} sx={{ p: 3, mb: 3, backgroundColor: 'success.light', color: 'success.contrastText' }}>
        <Typography variant="h6" gutterBottom>
          <CheckCircleIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          What Was Accomplished
        </Typography>
        
        <List>
          <ListItem>
            <ListItemIcon>
              <CheckCircleIcon sx={{ color: 'success.contrastText' }} />
            </ListItemIcon>
            <ListItemText 
              primary={`Configuration saved as "${filename}"`}
              secondary="Your configuration file has been successfully saved to the system"
              sx={{ color: 'success.contrastText' }}
            />
          </ListItem>
          
          {setAsCurrentConfig && (
            <ListItem>
              <ListItemIcon>
                <SettingsIcon sx={{ color: 'success.contrastText' }} />
              </ListItemIcon>
              <ListItemText 
                primary="Set as Active Configuration"
                secondary="This configuration is now the active setup for your UC2 system"
                sx={{ color: 'success.contrastText' }}
              />
            </ListItem>
          )}
          
          {restartAfterSave && (
            <ListItem>
              <ListItemIcon>
                <RestartIcon sx={{ color: 'success.contrastText' }} />
              </ListItemIcon>
              <ListItemText 
                primary="System Restart Completed"
                secondary="Your UC2 system has been restarted and is running with the new configuration"
                sx={{ color: 'success.contrastText' }}
              />
            </ListItem>
          )}
        </List>
      </Paper>

      {/* Configuration Details */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} color="info" />
          Configuration Details
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
          <Chip 
            label={`File: ${filename}`} 
            color="primary" 
            variant="outlined"
          />
          <Chip 
            label={setAsCurrentConfig ? "Active Configuration" : "Saved Configuration"} 
            color={setAsCurrentConfig ? "success" : "default"}
          />
          <Chip 
            label={restartAfterSave ? "System Restarted" : "No Restart"} 
            color={restartAfterSave ? "warning" : "default"}
          />
        </Box>
        
        <Typography variant="body2" color="text.secondary">
          Your configuration is now ready to use. You can access all UC2 system features 
          through the main interface with your new settings.
        </Typography>
      </Paper>

      {/* Next Steps */}
      <Paper elevation={1} sx={{ p: 3, mb: 3, backgroundColor: 'secondary' }}>
        <Typography variant="h6" gutterBottom>
          <PlayArrowIcon sx={{ mr: 1, verticalAlign: 'middle' }} color="primary" />
          What's Next?
        </Typography>
        
        <List dense>
          <ListItem>
            <ListItemText 
              primary="Test Your Configuration"
              secondary="Navigate to other tabs to test your actuators, detectors, and positioners"
            />
          </ListItem>
          
          <ListItem>
            <ListItemText 
              primary="Run System Diagnostics"
              secondary="Use the connection status indicators to verify all devices are working properly"
            />
          </ListItem>
          
          <ListItem>
            <ListItemText 
              primary="Create Additional Configurations"
              secondary="You can run this wizard again to create different setups for various experiments"
            />
          </ListItem>
          
          <ListItem>
            <ListItemText 
              primary="Backup Your Configuration"
              secondary="Consider backing up your configuration files for future use or sharing"
            />
          </ListItem>
        </List>
      </Paper>

      {/* Action Buttons */}
      <Paper elevation={1} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Choose Your Next Action
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<RefreshIcon />}
            onClick={onStartNewWizard}
            size="large"
          >
            Create Another Configuration
          </Button>
          
          <Button
            variant="outlined"
            color="primary"
            onClick={onComplete}
            size="large"
          >
            Return to UC2 Controller
          </Button>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
          You can always access this wizard again from the Configuration Editor tab.
        </Typography>
      </Paper>

      {/* Tips */}
      <Alert severity="info" sx={{ mt: 3 }}>
        <Typography variant="body2">
          <strong>Pro Tip:</strong> You can switch between different configurations anytime using 
          the "Select Setup" tab without needing to restart the system. Only restart when you've 
          made significant changes that require hardware reinitialization.
        </Typography>
      </Alert>
    </Box>
  );
};

export default ConfigWizardStep6;