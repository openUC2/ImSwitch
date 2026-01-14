import React from "react";
import {
  Box,
  Button,
  Typography,
  Paper,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Grid,
  Card,
  CardContent,
} from "@mui/material";
import {
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  CenterFocusStrong as CenterIcon,
  Search as SearchIcon,
  TouchApp as TouchIcon,
  Settings as SettingsIcon,
} from "@mui/icons-material";
import { useTheme } from '@mui/material/styles';
import LiveStreamTile from "../LiveStreamTile";

const StageCenterStep1 = ({ onNext, activeStep, totalSteps }) => {
  const theme = useTheme();
  const placeholderImageStyle = {
    width: "100%",
    height: "200px",
    backgroundColor: theme.palette.background.paper,
    border: "2px dashed #ccc",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: "16px",
  };

  return (
    <Box sx={{ maxWidth: 1000, mx: "auto", p: 2 }}>
      {/* Live Stream Tile - positioned in top right */}
      <Box sx={{ position: 'absolute', top: 20, right: 20, zIndex: 10 }}>
        <LiveStreamTile width={200} height={150} />
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Welcome to the Stage Center Calibration Wizard
        </Typography>
        This advanced wizard will help you find and calibrate the center position of your microscope stage 
        using both manual methods and automatic bright spot detection. Monitor the live stream during calibration.
      </Alert>

      <Paper elevation={2} sx={{ p: 3, mb: 3, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
        <Typography variant="h6" gutterBottom color="primary">
          <WarningIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          What This Wizard Will Do
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card elevation={1} sx={{ height: '100%', background: theme.palette.background.paper, color: theme.palette.text.primary }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle', color: '#1976d2' }} />
                  Manual Calibration
                </Typography>
                <Typography variant="body2">
                  • Enter known center positions manually<br/>
                  • Use current position as reference<br/>
                  • Apply offset corrections
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card elevation={1} sx={{ height: '100%', background: theme.palette.background.paper, color: theme.palette.text.primary }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  <SearchIcon sx={{ mr: 1, verticalAlign: 'middle', color: '#4caf50' }} />
                  Automatic Detection
                </Typography>
                <Typography variant="body2">
                  • Spiral scan to find bright spots<br/>
                  • Configurable search parameters<br/>
                  • Real-time position tracking
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card elevation={1} sx={{ height: '100%', background: theme.palette.background.paper, color: theme.palette.text.primary }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  <TouchIcon sx={{ mr: 1, verticalAlign: 'middle', color: '#ff9800' }} />
                  Interactive Stage Map
                </Typography>
                <Typography variant="body2">
                  • Visual stage representation<br/>
                  • Click-to-move functionality<br/>
                  • Real-time position display
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card elevation={1} sx={{ height: '100%', background: theme.palette.background.paper, color: theme.palette.text.primary }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  <CenterIcon sx={{ mr: 1, verticalAlign: 'middle', color: '#9c27b0' }} />
                  Precise Results
                </Typography>
                <Typography variant="body2">
                  • Accurate center calculation<br/>
                  • Multiple validation methods<br/>
                  • Easy result application
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      <Paper elevation={1} sx={{ p: 3, mb: 3, background: theme.palette.background.paper, color: theme.palette.text.primary }}>
        <Typography variant="h6" gutterBottom color="primary">
          <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} color="info" />
          Wizard Steps Overview
        </Typography>
        
        <List>
          <ListItem>
            <ListItemIcon>
              <CheckIcon color={activeStep >= 0 ? "success" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary="Step 1: Setup & Overview (Current)"
              secondary="Introduction to the calibration process and requirements"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color={activeStep >= 1 ? "success" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary="Step 2: Manual Position Entry"
              secondary="Enter known positions or use current stage position"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color={activeStep >= 2 ? "success" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary="Step 3: Stage Map Visualization"
              secondary="Interactive stage map with click-to-move functionality"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color={activeStep >= 3 ? "success" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary="Step 4: Automatic Detection"
              secondary="Automated spiral scan to find bright spots on the stage"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color={activeStep >= 4 ? "success" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary="Step 5: Review Results"
              secondary="Review and validate the calibration results"
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <CheckIcon color={activeStep >= 5 ? "success" : "disabled"} />
            </ListItemIcon>
            <ListItemText 
              primary="Step 6: Complete"
              secondary="Apply the calibration and finish the process"
            />
          </ListItem>
        </List>
      </Paper>

      <Alert severity="warning" sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          <strong>Important Notes:</strong>
        </Typography>
        <Typography variant="body2">
          • Ensure your stage is properly connected and responsive<br/>
          • For automatic detection, ensure there are bright spots or samples on the stage<br/>
          • The wizard will guide you through each step with clear instructions<br/>
          • You can navigate between steps using the stepper above
        </Typography>
      </Alert>

      <Box sx={{...placeholderImageStyle}}>
        <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center' }}>
          🔬 Stage Center Calibration Overview
          <br />
          <small>(Reference diagram showing stage coordinate system and center calibration concept)</small>
        </Typography>
      </Box>

      <Box sx={{ display: "flex", justifyContent: "space-between", mt: 3 }}>
        <Box /> {/* Placeholder for back button */}
        <Button 
          variant="contained" 
          onClick={onNext}
          size="large"
          sx={{
            background: theme.palette.primary.main,
            color: theme.palette.primary.contrastText,
            '&:hover': {
              background: theme.palette.primary.dark,
            }
          }}
        >
          Start Calibration
        </Button>
      </Box>
    </Box>
  );
};

export default StageCenterStep1;