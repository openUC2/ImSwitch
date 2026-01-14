import React from "react";
import { useSelector } from "react-redux";
import {
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import {
  CheckCircle as CheckIcon,
  Celebration as CelebrationIcon,
  RestartAlt as RestartIcon,
} from "@mui/icons-material";
import * as objectiveSlice from "../../state/slices/ObjectiveSlice.js";

const WizardStep6 = ({ hostIP, hostPort, onNext, onBack, onComplete, activeStep, totalSteps }) => {
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);

  const calibrationComplete = 
    objectiveState.posX0 !== null && 
    objectiveState.posX1 !== null && 
    objectiveState.posZ0 !== null && 
    objectiveState.posZ1 !== null;

  const handleRestartCalibration = () => {
    // Reset to step 1
    if (onBack) {
      // Go back to step 1 (index 0)
      for (let i = 0; i < 5; i++) {
        onBack();
      }
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <CelebrationIcon sx={{ fontSize: 40, color: 'success.main', mr: 2 }} />
        <Typography variant="h4" color="success.main">
          Calibration Complete!
        </Typography>
      </Box>
      
      {calibrationComplete ? (
        <Alert severity="success" sx={{ mb: 3 }}>
          🎉 Congratulations! You have successfully calibrated all objective positions.
          Your microscope is now ready for use with automatic objective switching and focus.
        </Alert>
      ) : (
        <Alert severity="warning" sx={{ mb: 3 }}>
          ⚠️ Some calibration values may be missing. Please review the summary below
          and consider re-running the calibration for any missing values.
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              <CheckIcon sx={{ mr: 1, verticalAlign: 'middle', color: 'success.main' }} />
              Calibration Summary
            </Typography>
            
            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color={objectiveState.posX0 !== null ? "success" : "disabled"} />
                </ListItemIcon>
                <ListItemText 
                  primary="X0 Position (Objective Slot 1)"
                  secondary={`Value: ${objectiveState.posX0 !== null ? objectiveState.posX0 : "Not calibrated"}`}
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color={objectiveState.posX1 !== null ? "success" : "disabled"} />
                </ListItemIcon>
                <ListItemText 
                  primary="X1 Position (Objective Slot 2)"
                  secondary={`Value: ${objectiveState.posX1 !== null ? objectiveState.posX1 : "Not calibrated"}`}
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color={objectiveState.posZ0 !== null ? "success" : "disabled"} />
                </ListItemIcon>
                <ListItemText 
                  primary="Z0 Position (Focus for Objective 1)"
                  secondary={`Value: ${objectiveState.posZ0 !== null ? objectiveState.posZ0 : "Not calibrated"}`}
                />
              </ListItem>
              
              <ListItem>
                <ListItemIcon>
                  <CheckIcon color={objectiveState.posZ1 !== null ? "success" : "disabled"} />
                </ListItemIcon>
                <ListItemText 
                  primary="Z1 Position (Focus for Objective 2)"
                  secondary={`Value: ${objectiveState.posZ1 !== null ? objectiveState.posZ1 : "Not calibrated"}`}
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              What's Next?
            </Typography>
            
            <Typography variant="body1" paragraph>
              Your objective calibration is now complete. You can:
            </Typography>
            
            <List>
              <ListItem>
                <ListItemText 
                  primary="• Use automatic objective switching"
                  secondary="The system will now automatically move to the correct position when switching objectives"
                />
              </ListItem>
              
              <ListItem>
                <ListItemText 
                  primary="• Benefit from automated focus"
                  secondary="The system will automatically adjust focus when switching between objectives"
                />
              </ListItem>
              
              <ListItem>
                <ListItemText 
                  primary="• Return to normal operation"
                  secondary="Close this wizard to return to the main objective controller interface"
                />
              </ListItem>
            </List>

            <Box sx={{ mt: 3, display: 'flex', gap: 2, flexDirection: 'column' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={onComplete}
                size="large"
                fullWidth
              >
                Return to Main Controller
              </Button>
              
              {!calibrationComplete && (
                <Button
                  variant="outlined"
                  color="secondary"
                  onClick={handleRestartCalibration}
                  startIcon={<RestartIcon />}
                  fullWidth
                >
                  Restart Calibration
                </Button>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      <Paper elevation={1} sx={{ p: 2, mt: 3, backgroundColor: '#f0f7ff' }}>
        <Typography variant="h6" gutterBottom color="primary">
          💡 Tips for Best Results
        </Typography>
        <Typography variant="body2" paragraph>
          • <strong>Regular Recalibration:</strong> Consider re-running this calibration if you change objectives or notice focus drift.
        </Typography>
        <Typography variant="body2" paragraph>
          • <strong>Fine Adjustments:</strong> You can still make manual adjustments in the main controller if needed.
        </Typography>
        <Typography variant="body2">
          • <strong>Backup Settings:</strong> Consider noting down these calibration values for future reference.
        </Typography>
      </Paper>
    </Box>
  );
};

export default WizardStep6;