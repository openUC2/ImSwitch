import React from "react";
import {
  Typography,
  Box,
  Paper,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Alert,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useTheme,
} from "@mui/material";
import {
  Settings as SettingsIcon,
  FolderOpen as FolderIcon,
  Add as AddIcon,
  Info as InfoIcon,
} from "@mui/icons-material";

const ConfigWizardStep1 = ({ 
  selectedSource, 
  onSourceChange, 
  currentActiveFilename,
  onNext, 
  onBack, 
  activeStep, 
  totalSteps 
}) => {
  const theme = useTheme();

  const handleSourceChange = (event) => {
    onSourceChange(event.target.value);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Welcome to the Configuration Wizard
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        This wizard will guide you through managing your UC2 configuration files step by step.
        You can load existing configurations, modify them, and save them with proper validation.
      </Alert>

      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          <SettingsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Choose Your Starting Point
        </Typography>
        
        <FormControl component="fieldset" sx={{ width: '100%' }}>
          <FormLabel component="legend" sx={{ mb: 2 }}>
            Select how you want to begin configuring your UC2 system:
          </FormLabel>
          
          <RadioGroup
            value={selectedSource}
            onChange={handleSourceChange}
            sx={{ gap: 2 }}
          >
            <Card 
              variant="outlined" 
              sx={{ 
                cursor: 'pointer',
                '&:hover': { backgroundColor: 'action.hover' },
                backgroundColor: selectedSource === 'current' ? 'action.selected' : 'transparent'
              }}
              onClick={() => onSourceChange('current')}
            >
              <CardContent>
                <FormControlLabel
                  value="current"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center' }}>
                        <SettingsIcon sx={{ mr: 1 }} color="primary" />
                        Load Current Active Configuration
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Start with the configuration that's currently running in your UC2 system.
                        This is the safest option if you want to make small modifications.
                        {currentActiveFilename && (
                          <>
                            <br />
                            <strong>Current file:</strong>{' '}
                            {typeof currentActiveFilename === 'object' ? (
                              <pre style={{
                                display: 'inline-block',
                                fontSize: 12,
                                background: theme.palette.mode === 'dark' ? '#222' : '#fff',
                                color: theme.palette.text.primary,
                                padding: 4,
                                borderRadius: 4,
                                margin: 0
                              }}>{JSON.stringify(currentActiveFilename, null, 2)}</pre>
                            ) : (
                              <span>{currentActiveFilename}</span>
                            )}
                          </>
                        )}
                      </Typography>
                    </Box>
                  }
                  sx={{ margin: 0, width: '100%' }}
                />
              </CardContent>
            </Card>

            <Card 
              variant="outlined" 
              sx={{ 
                cursor: 'pointer',
                '&:hover': { backgroundColor: 'action.hover' },
                backgroundColor: selectedSource === 'existing' ? 'action.selected' : 'transparent'
              }}
              onClick={() => onSourceChange('existing')}
            >
              <CardContent>
                <FormControlLabel
                  value="existing"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center' }}>
                        <FolderIcon sx={{ mr: 1 }} color="primary" />
                        Select from Available Configurations
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Choose from your existing configuration files. You can browse through
                        all available setups and load any one as your starting point.
                      </Typography>
                    </Box>
                  }
                  sx={{ margin: 0, width: '100%' }}
                />
              </CardContent>
            </Card>

            <Card 
              variant="outlined" 
              sx={{ 
                cursor: 'pointer',
                '&:hover': { backgroundColor: 'action.hover' },
                backgroundColor: selectedSource === 'new' ? 'action.selected' : 'transparent'
              }}
              onClick={() => onSourceChange('new')}
            >
              <CardContent>
                <FormControlLabel
                  value="new"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center' }}>
                        <AddIcon sx={{ mr: 1 }} color="primary" />
                        Create New Configuration
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Start with a blank configuration template. Choose this option if you want
                        to build a completely new setup from scratch.
                      </Typography>
                    </Box>
                  }
                  sx={{ margin: 0, width: '100%' }}
                />
              </CardContent>
            </Card>
          </RadioGroup>
        </FormControl>
      </Paper>

      <Paper elevation={1} sx={{ p: 2, backgroundColor: 'black' }}>
        <Typography variant="h6" gutterBottom>
          <InfoIcon sx={{ mr: 1, verticalAlign: 'middle' }} color="info" />
          What Happens Next
        </Typography>
        
        <List dense>
          <ListItem>
            <ListItemIcon>
              <Typography variant="body2" color="primary" fontWeight="bold">2</Typography>
            </ListItemIcon>
            <ListItemText 
              primary="Load Configuration" 
              secondary="Load and preview your selected configuration"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Typography variant="body2" color="primary" fontWeight="bold">3</Typography>
            </ListItemIcon>
            <ListItemText 
              primary="Edit & Validate" 
              secondary="Modify the configuration with built-in validation"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Typography variant="body2" color="primary" fontWeight="bold">4</Typography>
            </ListItemIcon>
            <ListItemText 
              primary="Save Options" 
              secondary="Choose filename, activation, and restart preferences"
            />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <Typography variant="body2" color="primary" fontWeight="bold">5</Typography>
            </ListItemIcon>
            <ListItemText 
              primary="Complete" 
              secondary="Apply your configuration and monitor the system"
            />
          </ListItem>
        </List>
      </Paper>

      {!selectedSource && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Please select one of the options above to continue.
        </Alert>
      )}
    </Box>
  );
};

export default ConfigWizardStep1;