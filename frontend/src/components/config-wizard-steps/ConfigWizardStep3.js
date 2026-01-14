import React, { useState, useCallback } from "react";
import {
  Typography,
  Box,
  Paper,
  Button,
  Alert,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
} from "@mui/material";
import {
  Edit as EditIcon,
  Preview as PreviewIcon,
  Check as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from "@mui/icons-material";
import { JsonEditor } from "json-edit-react";
import AceEditor from "react-ace";
import "ace-builds/src-noconflict/mode-json";
import "ace-builds/src-noconflict/theme-github";
import 'ace-builds/src-noconflict/worker-json';
import { 
  validateConfiguration, 
  validateJsonString, 
  createConfigurationPreview 
} from "../../utils/configValidation";

const ConfigWizardStep3 = ({ 
  loadedConfig,
  editorJson,
  editorJsonText,
  useAceEditor,
  onConfigChange,
  onJsonTextChange,
  onEditorTypeChange,
  onNext, 
  onBack, 
  activeStep, 
  totalSteps 
}) => {
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [previewData, setPreviewData] = useState(null);

  const handleValidate = useCallback(() => {
    setIsValidating(true);
    
    // Simulate async validation (in case we want to add server-side validation later)
    setTimeout(() => {
      let configToValidate = null;
      
      if (useAceEditor) {
        if (!editorJsonText?.trim()) {
          setValidationResult({
            isValid: false,
            errors: ['No JSON content to validate'],
            warnings: []
          });
          setIsValidating(false);
          return;
        }
        
        const jsonValidation = validateJsonString(editorJsonText);
        if (!jsonValidation.isValid) {
          setValidationResult({
            isValid: false,
            errors: [`Invalid JSON: ${jsonValidation.error}`],
            warnings: []
          });
          setIsValidating(false);
          return;
        }
        configToValidate = jsonValidation.parsed;
      } else {
        configToValidate = editorJson || loadedConfig;
      }
      
      if (!configToValidate) {
        setValidationResult({
          isValid: false,
          errors: ['No configuration to validate'],
          warnings: []
        });
        setIsValidating(false);
        return;
      }
      
      const result = validateConfiguration(configToValidate);
      setValidationResult(result);
      setIsValidating(false);
    }, 500);
  }, [useAceEditor, editorJsonText, editorJson, loadedConfig]);

  const handlePreview = useCallback(() => {
    let configToPreview = null;
    
    if (useAceEditor) {
      if (!editorJsonText?.trim()) return;
      const jsonValidation = validateJsonString(editorJsonText);
      if (!jsonValidation.isValid) return;
      configToPreview = jsonValidation.parsed;
    } else {
      configToPreview = editorJson || loadedConfig;
    }
    
    if (configToPreview) {
      const preview = createConfigurationPreview(configToPreview);
      setPreviewData(preview);
      setShowPreview(true);
    }
  }, [useAceEditor, editorJsonText, editorJson, loadedConfig]);

  const getCurrentConfig = () => {
    if (useAceEditor) {
      return editorJsonText;
    }
    return editorJson || loadedConfig;
  };

  const hasValidConfig = () => {
    const config = getCurrentConfig();
    if (useAceEditor) {
      return config && config.trim() !== '';
    }
    return config && typeof config === 'object';
  };

  const getValidationSummary = () => {
    if (!validationResult) return null;
    
    const { isValid, errors = [], warnings = [] } = validationResult;
    const errorCount = errors && Array.isArray(errors) ? errors.length : 0;
    const warningCount = warnings && Array.isArray(warnings) ? warnings.length : 0;
    const totalIssues = errorCount + warningCount;
    
    if (isValid && totalIssues === 0) {
      return { type: 'success', message: 'Configuration is valid with no issues' };
    } else if (errorCount === 0) {
      return { type: 'warning', message: `Configuration is valid but has ${warningCount} warning${warningCount !== 1 ? 's' : ''}` };
    } else {
      return { type: 'error', message: `Configuration has ${errorCount} error${errorCount !== 1 ? 's' : ''}${warningCount > 0 ? ` and ${warningCount} warning${warningCount !== 1 ? 's' : ''}` : ''}` };
    }
  };

  const summary = getValidationSummary();

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Edit & Validate Configuration
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>Modify your configuration using the editor below.</strong> Use the validation feature to check for issues before proceeding to the save step.
        </Typography>
        <Typography variant="body2">
          💡 <strong>Tip:</strong> Use the Visual Editor for guided editing, or switch to Text Editor for direct JSON manipulation. 
          The Text Editor supports formatting (Ctrl+Alt+F) and has better copy/paste support.
        </Typography>
      </Alert>

      {/* Editor Type Toggle */}
      <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h6">
            <EditIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Configuration Editor
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={useAceEditor}
                onChange={(e) => onEditorTypeChange(e.target.checked)}
              />
            }
            label={`${useAceEditor ? 'Text' : 'Visual'} Editor`}
          />
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {useAceEditor 
            ? "Edit JSON directly with syntax highlighting and validation. Use this for advanced editing or copy/paste operations."
            : "Use the visual editor for a guided approach to editing configuration values. Best for beginners."
          }
        </Typography>
      </Paper>

      {/* Validation Status */}
      {isValidating && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Validating configuration...
          </Typography>
        </Box>
      )}

      {summary && (
        <Alert 
          severity={summary.type} 
          sx={{ mb: 3 }}
          action={
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button 
                color="inherit" 
                size="small" 
                onClick={handlePreview}
                disabled={!hasValidConfig()}
              >
                Preview
              </Button>
              <Button 
                color="inherit" 
                size="small" 
                onClick={handleValidate}
                disabled={isValidating}
              >
                Re-validate
              </Button>
            </Box>
          }
        >
          {summary.message}
        </Alert>
      )}

      {/* Validation Issues */}
      {validationResult && (
        (validationResult.errors && Array.isArray(validationResult.errors) && validationResult.errors.length > 0) || 
        (validationResult.warnings && Array.isArray(validationResult.warnings) && validationResult.warnings.length > 0)
      ) && (
        <Paper elevation={1} sx={{ p: 2, mb: 3 }}>
          {validationResult.errors && Array.isArray(validationResult.errors) && validationResult.errors.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="error" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <ErrorIcon sx={{ mr: 1 }} />
                Errors ({validationResult.errors.length})
              </Typography>
              {validationResult.errors.map((error, index) => (
                <Typography key={index} variant="body2" color="error" sx={{ ml: 3, mb: 0.5 }}>
                  • {error}
                </Typography>
              ))}
            </Box>
          )}
          
          {validationResult.warnings && Array.isArray(validationResult.warnings) && validationResult.warnings.length > 0 && (
            <Box>
              <Typography variant="subtitle2" color="warning.main" sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <WarningIcon sx={{ mr: 1 }} />
                Warnings ({validationResult.warnings.length})
              </Typography>
              {validationResult.warnings.map((warning, index) => (
                <Typography key={index} variant="body2" color="warning.main" sx={{ ml: 3, mb: 0.5 }}>
                  • {warning}
                </Typography>
              ))}
            </Box>
          )}
        </Paper>
      )}

      {/* Editor */}
      <Paper elevation={2} sx={{ mb: 3 }}>
        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">
              {useAceEditor ? 'JSON Text Editor' : 'Visual Editor'}
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              {useAceEditor && (
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => {
                    try {
                      const parsed = JSON.parse(editorJsonText || '{}');
                      const formatted = JSON.stringify(parsed, null, 2);
                      onJsonTextChange(formatted);
                    } catch (e) {
                      // Show error if JSON is invalid
                      alert('Unable to format: Invalid JSON syntax');
                    }
                  }}
                  disabled={!editorJsonText?.trim()}
                >
                  Format JSON
                </Button>
              )}
              <Button
                variant="outlined"
                size="small"
                startIcon={<PreviewIcon />}
                onClick={handlePreview}
                disabled={!hasValidConfig()}
              >
                Preview
              </Button>
              <Button
                variant="outlined"
                size="small"
                startIcon={<CheckIcon />}
                onClick={handleValidate}
                disabled={isValidating || !hasValidConfig()}
              >
                Validate
              </Button>
            </Box>
          </Box>
          {useAceEditor && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Use Ctrl+Alt+F (Cmd+Alt+F on Mac) to format JSON, or click the Format JSON button above
            </Typography>
          )}
        </Box>
        
        <Box sx={{ minHeight: '400px' }}>
          {useAceEditor ? (
            <AceEditor
              mode="json"
              theme="github"
              onChange={onJsonTextChange}
              value={editorJsonText || ''}
              name="configEditor"
              editorProps={{ $blockScrolling: true }}
              width="100%"
              height="400px"
              fontSize={14}
              showPrintMargin={true}
              showGutter={true}
              highlightActiveLine={true}
              setOptions={{
                useWorker: false, // Disable worker to prevent network error
                enableBasicAutocompletion: true,
                enableLiveAutocompletion: false, // Disable to reduce noise
                enableSnippets: false, // Disable snippets for cleaner experience
                showLineNumbers: true,
                tabSize: 2,
              }}
            />
          ) : (
            <Box sx={{ height: '400px', overflow: 'auto', p: 1 }}>
              {(editorJson || loadedConfig) ? (
                <JsonEditor
                  data={editorJson || loadedConfig}
                  setData={onConfigChange}
                  restrictEdit={false}
                  restrictDelete={false}
                  restrictAdd={false}
                  indent={2}
                  collapse={false}
                  enableClipboard={true}
                  displayDataTypes={true}
                  displayObjectSize={true}
                  displayArrayKey={true}
                />
              ) : (
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  height: '100%',
                  color: 'text.secondary',
                  flexDirection: 'column',
                  gap: 2
                }}>
                  <Typography variant="h6">No configuration loaded</Typography>
                  <Typography variant="body2">
                    Go back to Step 2 to load a configuration file first.
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </Box>
      </Paper>

      {!hasValidConfig() && (
        <Alert severity="warning">
          Please ensure you have a valid configuration before proceeding to the next step.
        </Alert>
      )}

      {/* Preview Dialog */}
      <Dialog
        open={showPreview}
        onClose={() => setShowPreview(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Configuration Preview</DialogTitle>
        <DialogContent>
          {previewData && previewData.valid ? (
            <Box>
              <Typography variant="h6" gutterBottom>Device Summary</Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                <Chip label={`${previewData.detectorCount || 0} Detectors`} color="primary" />
                {(previewData.laserCount || 0) > 0 && (
                  <Chip label={`${previewData.laserCount} Lasers`} color="secondary" />
                )}
                {(previewData.actuatorCount || 0) > 0 && (
                  <Chip label={`${previewData.actuatorCount} Actuators`} color="primary" />
                )}
                <Chip label={`${previewData.positionerCount || 0} Positioners`} color="primary" />
                <Chip label={`${previewData.totalDevices || 0} Total`} variant="outlined" />
              </Box>
              
              {previewData.devices && Array.isArray(previewData.devices) && previewData.devices.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>Configured Devices</Typography>
                  {previewData.devices.map((device, index) => (
                    <Typography key={index} variant="body2" sx={{ ml: 2 }}>
                      • <strong>{device.name}</strong> ({device.type})
                      {device.manager && ` - ${device.manager}`}
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              {previewData ? previewData.summary : 'Unable to generate preview for this configuration.'}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPreview(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ConfigWizardStep3;