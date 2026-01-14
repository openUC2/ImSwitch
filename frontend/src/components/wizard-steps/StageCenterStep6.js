import React from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  Box,
  Button,
  Typography,
  Paper,
  Alert,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from "@mui/material";
import {
  CheckCircle as CheckIcon,
  Celebration as CelebrationIcon,
  Refresh as RefreshIcon,
  Home as HomeIcon,
  Save as SaveIcon,
} from "@mui/icons-material";
import * as stageCenterCalibrationSlice from "../../state/slices/StageCenterCalibrationSlice";
import LiveStreamTile from "../LiveStreamTile";
import { useTheme } from '@mui/material/styles';

const StageCenterStep6 = ({ onComplete, activeStep, totalSteps }) => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const stageCenterState = useSelector(stageCenterCalibrationSlice.getStageCenterCalibrationState);
  
  const {
    currentX,
    currentY,
    manualCenterX,
    manualCenterY,
    foundCenterX,
    foundCenterY,
    calibrationResults,
  } = stageCenterState;

  const hasManualCenter = manualCenterX && manualCenterY;
  const hasFoundCenter = foundCenterX !== null && foundCenterY !== null;
  const hasCalibrationData = calibrationResults.length > 0;

  const resetCalibration = () => {
    dispatch(stageCenterCalibrationSlice.resetCalibrationResults());
    dispatch(stageCenterCalibrationSlice.setActiveStep(0));
  };

  const completeAndClose = () => {
    dispatch(stageCenterCalibrationSlice.resetWizard());
    onComplete();
  };

  const saveCalibrationSummary = () => {
    const summary = {
      timestamp: new Date().toISOString(),
      currentPosition: { x: currentX, y: currentY },
      manualCenter: hasManualCenter ? { x: parseFloat(manualCenterX), y: parseFloat(manualCenterY) } : null,
      foundCenter: hasFoundCenter ? { x: foundCenterX, y: foundCenterY } : null,
      scanResults: calibrationResults,
      scanCount: calibrationResults.length,
    };

    const dataStr = JSON.stringify(summary, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `stage_center_calibration_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box sx={{ maxWidth: 1000, mx: "auto", p: 2 }}>
      {/* Live Stream Tile - positioned in top right */}
      <Box sx={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
        <LiveStreamTile width={200} height={150} />
      </Box>

      <Alert severity="success" sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <CelebrationIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Stage Center Calibration Complete!
        </Typography>
        Your stage center calibration has been completed successfully. Review the summary below 
        and choose your next steps. The live stream shows your final position.
      </Alert>

      <Grid container spacing={3}>
        {/* Calibration Summary */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                Calibration Summary
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckIcon color={hasManualCenter ? "success" : "disabled"} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Manual Center Entry"
                    secondary={hasManualCenter 
                      ? `Position: (${parseFloat(manualCenterX).toFixed(2)}, ${parseFloat(manualCenterY).toFixed(2)}) μm`
                      : "No manual center position entered"
                    }
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckIcon color={hasFoundCenter ? "success" : "disabled"} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Automatic Detection"
                    secondary={hasFoundCenter 
                      ? `Found center: (${foundCenterX.toFixed(2)}, ${foundCenterY.toFixed(2)}) μm`
                      : "No automatic detection performed"
                    }
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <CheckIcon color={hasCalibrationData ? "success" : "disabled"} />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Scan Data"
                    secondary={hasCalibrationData 
                      ? `${calibrationResults.length} positions scanned`
                      : "No scan data available"
                    }
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Next Steps */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ background: theme.palette.background.paper, color: theme.palette.text.primary }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                What's Next?
              </Typography>
              
              <Typography variant="body2" paragraph>
                Your calibration data has been collected and is available for use. You can:
              </Typography>
              
              <List dense>
                <ListItem>
                  <ListItemIcon>
                    <SaveIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Apply Center Positions"
                    secondary="Use the positions found in Step 5 to set your stage offsets"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <RefreshIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Repeat Calibration"
                    secondary="Run the wizard again with different parameters if needed"
                  />
                </ListItem>
                
                <ListItem>
                  <ListItemIcon>
                    <HomeIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Return to Normal Operation"
                    secondary="Close the wizard and continue with your microscopy work"
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Divider sx={{ my: 3 }} />

      {/* Action Buttons */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Button
            variant="outlined"
            onClick={saveCalibrationSummary}
            fullWidth
            startIcon={<SaveIcon />}
            disabled={!hasManualCenter && !hasFoundCenter && !hasCalibrationData}
          >
            Download Summary
          </Button>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Button
            variant="outlined"
            onClick={resetCalibration}
            fullWidth
            startIcon={<RefreshIcon />}
          >
            Start Over
          </Button>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Button
            variant="contained"
            onClick={completeAndClose}
            fullWidth
            startIcon={<HomeIcon />}
            sx={{
              background: theme.palette.success.main,
              color: theme.palette.success.contrastText,
              '&:hover': {
                background: theme.palette.success.dark,
              }
            }}
          >
            Complete & Close
          </Button>
        </Grid>
      </Grid>

      <Paper elevation={1} sx={{ p: 3, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
        <Typography variant="h6" gutterBottom>
          Tips for Future Use
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Accuracy:</strong> The automatic detection works best when there are clear, bright 
          features on your stage. For subtle samples, manual positioning might be more accurate.
        </Typography>
        <Typography variant="body2" paragraph>
          <strong>Repeatability:</strong> Save your calibration parameters for future use. The same 
          scan settings can be applied to similar samples.
        </Typography>
        <Typography variant="body2">
          <strong>Validation:</strong> Always verify that the found center position makes sense for 
          your specific setup and sample type before applying it to important experiments.
        </Typography>
      </Paper>

      <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
        <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center' }}>
          Thank you for using the Stage Center Calibration Wizard!
          <br />
          Your feedback helps us improve the calibration process.
        </Typography>
      </Box>
    </Box>
  );
};

export default StageCenterStep6;