import React from "react";
import {
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
} from "@mui/material";
import {
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
} from "@mui/icons-material";

const WizardStep1 = ({ hostIP, hostPort, onNext, onBack, activeStep, totalSteps }) => {
  const placeholderImageStyle = {
    width: '100%',
    height: '300px',
    backgroundColor: '#f5f5f5',
    border: '2px dashed #ccc',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '8px',
    marginTop: '16px',
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Welcome to the Objective Calibration Wizard
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        This wizard will guide you through calibrating the objective positions for slots 1 and 2,
        including both the switching positions (X0, X1) and focus positions (Z0, Z1).
      </Alert>

      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom color="primary">
          <WarningIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Before You Begin - Setup Requirements
        </Typography>
        
        <List>
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Remove Objective Lenses"
              secondary="Carefully unscrew and remove both objective lenses from slots 1 and 2"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Remove Camera"
              secondary="Unscrew and remove the camera from the C-mount"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Insert Calibration Piece"
              secondary="Insert the calibration piece into the C-mount slot and light it up"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color="primary" />
            </ListItemIcon>
            <ListItemText 
              primary="Insert RMS Crosshair"
              secondary="Insert the RMS threaded crosshair into one of the objective slots"
            />
          </ListItem>
        </List>

        <Box sx={{placeholderImageStyle, backgroundColor: '#0000'}}>
          <Typography variant="body1" color="textSecondary">
            📷 Setup Reference Image Placeholder
            <br />
            <small>(Image showing calibration piece and crosshair setup will be displayed here)</small>
          </Typography>
        </Box>
      </Paper>

      <Paper elevation={1} sx={{ p: 2, backgroundColor: '#0000' }}>
        <Typography variant="h6" gutterBottom>
          <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} color="info" />
          What This Wizard Will Do
        </Typography>
        
        <Typography variant="body2" paragraph>
          • <strong>Steps 2-3:</strong> Calibrate the X0 and X1 positions (objective switching positions)
        </Typography>
        <Typography variant="body2" paragraph>
          • <strong>Steps 4-5:</strong> Calibrate the Z0 and Z1 positions (focus positions for each objective)
        </Typography>
        <Typography variant="body2">
          • <strong>Step 6:</strong> Complete the calibration and return to normal operation
        </Typography>
      </Paper>

      <Alert severity="warning" sx={{ mt: 3 }}>
        <strong>Important:</strong> Make sure you have completed all setup steps above before proceeding.
        The live view will be available in the next steps to help guide the calibration process.
      </Alert>
    </Box>
  );
};

export default WizardStep1;