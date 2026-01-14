import React from "react";
import {
  Typography,
  Box,
  Paper,
  TextField,
  FormControlLabel,
  Checkbox,
  Alert,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from "@mui/material";
import {
  Save as SaveIcon,
  Settings as SettingsIcon,
  RestartAlt as RestartIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
} from "@mui/icons-material";

const ConfigWizardStep4 = ({ 
  filename,
  setAsCurrentConfig,
  restartAfterSave,
  overwriteFile,
  onFilenameChange,
  onSetAsCurrentConfigChange,
  onRestartAfterSaveChange,
  onOverwriteFileChange,
  availableSetups,
  onNext, 
  onBack, 
  activeStep, 
  totalSteps 
}) => {
  const fileExists = filename && availableSetups.includes(filename);
  
  const getFilenameSuggestion = () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    return `config-${timestamp}.json`;
  };

  const getWarnings = () => {
    const warnings = [];
    
    if (!filename) {
      warnings.push('A filename is required to save the configuration');
    }

    // check if filename is a path or contains invalid characters
    if (filename && /[<>:"/\\|?*]/.test(filename)) {
      warnings.push('Filename contains invalid characters (<>:"/\\|?*)');
    }
    
    if (fileExists && !overwriteFile) {
      warnings.push(`File "${filename}" already exists. Enable "Overwrite if exists" to replace it.`);
    }
    
    if (restartAfterSave) {
      warnings.push('The system will restart after saving, which may take several minutes');
    }
    
    return warnings;
  };

  const warnings = getWarnings();
  const canProceed = filename && (!fileExists || overwriteFile);

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Save Configuration Options
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        Configure how your configuration will be saved and applied to the system.
      </Alert>

      {/* Filename Section */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <SaveIcon sx={{ mr: 1 }} color="primary" />
          File Settings
        </Typography>
        
        <TextField
          fullWidth
          label="Configuration Filename"
          value={filename}
          onChange={(e) => onFilenameChange(e.target.value)}
          placeholder={getFilenameSuggestion()}
          helperText={
            fileExists 
              ? `⚠️ File "${filename}" already exists` 
              : "Enter a name for your configuration file"
          }
          error={fileExists && !overwriteFile}
          sx={{ mb: 2 }}
        />
        
        <FormControlLabel
          control={
            <Checkbox
              checked={overwriteFile}
              onChange={(e) => onOverwriteFileChange(e.target.checked)}
              color="warning"
            />
          }
          label="Overwrite if file exists"
          sx={{ 
            color: overwriteFile ? 'warning.main' : 'text.primary',
            '& .MuiFormControlLabel-label': {
              fontWeight: overwriteFile ? 'bold' : 'normal'
            }
          }}
        />
        
        {fileExists && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="body2">
              The file "{filename}" already exists. 
              {overwriteFile 
                ? ' It will be replaced with your new configuration.' 
                : ' Please choose a different name or enable overwrite.'
              }
            </Typography>
          </Alert>
        )}
      </Paper>

      {/* System Settings Section */}
      <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <SettingsIcon sx={{ mr: 1 }} color="primary" />
          System Settings
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={setAsCurrentConfig}
                  onChange={(e) => onSetAsCurrentConfigChange(e.target.checked)}
                  color="primary"
                />
              }
              label="Set as Current Active Configuration"
            />
            <Typography variant="body2" color="text.secondary" sx={{ ml: 4, mt: 0.5 }}>
              This configuration will become the active one used by the system
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={restartAfterSave}
                  onChange={(e) => onRestartAfterSaveChange(e.target.checked)}
                  color="warning"
                />
              }
              label="Restart System After Save"
              sx={{ 
                color: restartAfterSave ? 'warning.main' : 'text.primary',
                '& .MuiFormControlLabel-label': {
                  fontWeight: restartAfterSave ? 'bold' : 'normal'
                }
              }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ ml: 4, mt: 0.5 }}>
              The UC2 system will restart to apply the new configuration
            </Typography>
          </Grid>
        </Grid>
        
        {(setAsCurrentConfig || restartAfterSave) && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              {setAsCurrentConfig && restartAfterSave && 
                "The configuration will be saved, set as active, and the system will restart automatically."
              }
              {setAsCurrentConfig && !restartAfterSave && 
                "The configuration will be saved and set as the active configuration, but the system won't restart automatically."
              }
              {!setAsCurrentConfig && restartAfterSave && 
                "The configuration will be saved and the system will restart, but this won't become the active configuration."
              }
            </Typography>
          </Alert>
        )}
      </Paper>

      {/* Configuration Summary */}
      <Paper elevation={1} sx={{ p: 3, mb: 3, backgroundColor: 'secondary' }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <InfoIcon sx={{ mr: 1 }} color="info" />
          Summary of Actions
        </Typography>
        
        <List dense>
          <ListItem>
            <ListItemIcon>
              <SaveIcon color={filename ? 'primary' : 'disabled'} />
            </ListItemIcon>
            <ListItemText 
              primary={`Save configuration as "${filename || '[filename required]'}"`}
              secondary={fileExists && overwriteFile ? 'Will overwrite existing file' : 'Will create new file'}
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <SettingsIcon color={setAsCurrentConfig ? 'primary' : 'disabled'} />
            </ListItemIcon>
            <ListItemText 
              primary={setAsCurrentConfig ? 'Set as active configuration' : 'Save without setting as active'}
              secondary={setAsCurrentConfig ? 'This will become the system\'s active configuration' : 'Configuration will be saved but not activated'}
            />
          </ListItem>
          
          <ListItem>
            <ListItemIcon>
              <RestartIcon color={restartAfterSave ? 'warning' : 'disabled'} />
            </ListItemIcon>
            <ListItemText 
              primary={restartAfterSave ? 'Restart system after save' : 'No system restart'}
              secondary={restartAfterSave ? 'System will restart automatically (may take several minutes)' : 'System will continue running with current configuration'}
            />
          </ListItem>
        </List>
      </Paper>

      {/* Warnings */}
      {warnings.length > 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body2" component="div">
            <strong>Please review the following:</strong>
            <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
              {warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          </Typography>
        </Alert>
      )}

      {canProceed && (
        <Alert severity="success" sx={{ mb: 3 }}>
          <Typography variant="body2">
            Configuration is ready to be saved. Click "Next" to proceed with the save operation.
          </Typography>
        </Alert>
      )}
    </Box>
  );
};

export default ConfigWizardStep4;