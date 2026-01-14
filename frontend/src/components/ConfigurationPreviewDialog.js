import React from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Paper,
  Box,
  Divider,
  Alert,
  Grid,
  Chip,
} from "@mui/material";
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from "@mui/icons-material";

const ConfigurationPreviewDialog = ({
  open,
  onClose,
  onConfirmSave,
  validationResult,
  configPreview,
  filename,
  isSaving = false,
}) => {
  const getValidationIcon = () => {
    if (!validationResult) return null;
    
    if (validationResult.isValid) {
      return <CheckCircleIcon color="success" />;
    } else {
      return <ErrorIcon color="error" />;
    }
  };

  const getValidationSeverity = () => {
    if (!validationResult) return "info";
    return validationResult.isValid ? "success" : "error";
  };

  const renderValidationMessage = () => {
    if (!validationResult) return null;

    const { isValid, errors, warnings } = validationResult;

    return (
      <Box>
        <Alert 
          severity={getValidationSeverity()} 
          icon={getValidationIcon()}
          sx={{ mb: 2 }}
        >
          <Typography variant="subtitle2">
            {isValid ? "Configuration is valid" : "Configuration has errors"}
          </Typography>
        </Alert>

        {errors.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="error" gutterBottom>
              Errors:
            </Typography>
            {errors.map((error, index) => (
              <Typography key={index} variant="body2" color="error" sx={{ ml: 2 }}>
                • {error}
              </Typography>
            ))}
          </Box>
        )}

        {warnings.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="warning.main" gutterBottom>
              Warnings:
            </Typography>
            {warnings.map((warning, index) => (
              <Typography key={index} variant="body2" color="warning.main" sx={{ ml: 2 }}>
                • {warning}
              </Typography>
            ))}
          </Box>
        )}
      </Box>
    );
  };

  const renderConfigPreview = () => {
    if (!configPreview || !configPreview.valid) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Configuration Summary
        </Typography>
        <Paper sx={{ p: 2, bgcolor: "grey.50" }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Detectors:
                </Typography>
                <Chip 
                  label={configPreview.detectors} 
                  size="small" 
                  color={configPreview.detectors > 0 ? "primary" : "default"}
                />
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Actuators:
                </Typography>
                <Chip 
                  label={configPreview.actuators} 
                  size="small" 
                  color={configPreview.actuators > 0 ? "primary" : "default"}
                />
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Positioners:
                </Typography>
                <Chip 
                  label={configPreview.positioners} 
                  size="small" 
                  color={configPreview.positioners > 0 ? "primary" : "default"}
                />
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  Lasers:
                </Typography>
                <Chip 
                  label={configPreview.lasers} 
                  size="small" 
                  color={configPreview.lasers > 0 ? "secondary" : "default"}
                />
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    );
  };

  const canSave = validationResult && validationResult.isValid;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: { minHeight: "400px" }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          {getValidationIcon()}
          <Typography variant="h6">
            Preview Configuration: {filename}
          </Typography>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {renderValidationMessage()}
        <Divider sx={{ my: 2 }} />
        {renderConfigPreview()}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={isSaving}>
          Cancel
        </Button>
        <Button 
          onClick={onConfirmSave} 
          variant="contained"
          disabled={!canSave || isSaving}
          startIcon={isSaving ? null : null}
        >
          {isSaving ? "Saving..." : "Save Configuration"}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfigurationPreviewDialog;