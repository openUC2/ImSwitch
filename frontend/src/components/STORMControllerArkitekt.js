// src/components/STORMControllerArkitekt.js
import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import createAxiosInstance from '../backendapi/createAxiosInstance';
import {
  Paper,
  Tabs,
  Tab,
  Box,
  Typography,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Checkbox,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Divider,
} from "@mui/material";
import { green, red } from "@mui/material/colors";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import { useWebSocket } from "../context/WebSocketContext";
import * as stormSlice from "../state/slices/STORMSlice.js";
import * as connectionSettingsSlice from "../state/slices/ConnectionSettingsSlice.js";
import LiveViewControlWrapper from "../axon/LiveViewControlWrapper";
import LiveViewSettings from "./LiveViewSettings";
import STORMPlot from "./STORMPlot";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`arkitekt-tabpanel-${index}`}
      aria-labelledby={`arkitekt-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
};

const STORMControllerArkitekt = () => {
  const dispatch = useDispatch();

  // Redux state
  const stormState = useSelector(stormSlice.getSTORMState);
  const connectionSettingsState = useSelector(
    connectionSettingsSlice.getConnectionSettingsState
  );

  // Local state
  const [activeStep, setActiveStep] = useState(0);
  const [reconstructionTabIndex, setReconstructionTabIndex] = useState(0);
  const [visualizationTabIndex, setVisualizationTabIndex] = useState(0);
  const [arkitektConnected, setArkitektConnected] = useState(false);
  const [availableWorkflows, setAvailableWorkflows] = useState([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState("");
  const [workflowStatus, setWorkflowStatus] = useState("idle");
  const [reconImage, setReconImage] = useState(null);
  const [plotData, setPlotData] = useState(null);
  const [statistics, setStatistics] = useState(null);

  const socket = useWebSocket();

  // Access Redux state
  const acquisitionParameters = stormState.acquisitionParameters;

  const steps = [
    "Connect to Arkitekt",
    "Select Workflow",
    "Configure Parameters",
    "Start Processing",
    "Monitor Results",
  ];

  // WebSocket signal handling for Arkitekt
  useEffect(() => {
    if (!socket) return;

    const handleSignal = (data) => {
      try {
        const jdata = typeof data === "string" ? JSON.parse(data) : data;

        // Handle Arkitekt-specific signals
        if (jdata.name === "sigArkitektConnectionStatus") {
          setArkitektConnected(jdata.connected);
        }

        if (jdata.name === "sigArkitektWorkflowUpdate") {
          setWorkflowStatus(jdata.status);
        }

        // Handle reconstruction results from Arkitekt
        if (jdata.name === "sigArkitektReconstructionResult" && jdata.image) {
          setReconImage(`data:image/jpeg;base64,${jdata.image}`);
        }

        if (
          jdata.name === "sigArkitektLocalizationResult" &&
          jdata.localizations
        ) {
          setPlotData(jdata.localizations);
        }

        if (jdata.name === "sigArkitektStatisticsResult" && jdata.statistics) {
          setStatistics(jdata.statistics);
        }
      } catch (error) {
        console.error("Error parsing Arkitekt signal data:", error);
      }
    };

    socket.on("signal", handleSignal);
    return () => socket.off("signal", handleSignal);
  }, [socket]);

  // Arkitekt operations
  const connectToArkitekt = async () => {
    try {
      const api = createAxiosInstance();
      const response = await api.get(
        `/ArkitektController/connect`
      );
      setArkitektConnected(true);
      await loadAvailableWorkflows();
    } catch (error) {
      console.error("Error connecting to Arkitekt:", error);
    }
  };

  const loadAvailableWorkflows = async () => {
    try {
      const api = createAxiosInstance();
      const response = await api.get(
        `/ArkitektController/getWorkflows`
      );
      const workflows = response.data;
      setAvailableWorkflows(workflows);
    } catch (error) {
      console.error("Error loading workflows:", error);
    }
  };

  const startArkitektProcessing = async () => {
    try {
      const api = createAxiosInstance();
      const response = await api.post(
        `/ArkitektController/startWorkflow`,
        {
          workflow: selectedWorkflow,
          parameters: acquisitionParameters,
        }
      );
      setWorkflowStatus("running");
    } catch (error) {
      console.error("Error starting Arkitekt processing:", error);
    }
  };

  const stopArkitektProcessing = async () => {
    try {
      const api = createAxiosInstance();
      const response = await api.post(
        `/ArkitektController/stopWorkflow`
      );
      setWorkflowStatus("stopped");
    } catch (error) {
      console.error("Error stopping Arkitekt processing:", error);
    }
  };

  const setAcquisitionParameter = (paramName, value) => {
    const updatedAcquisition = { ...acquisitionParameters, [paramName]: value };
    dispatch(stormSlice.setAcquisitionParameters(updatedAcquisition));
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleReconstructionTabChange = (event, newValue) => {
    setReconstructionTabIndex(newValue);
  };

  const handleVisualizationTabChange = (event, newValue) => {
    setVisualizationTabIndex(newValue);
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0: // Connect to Arkitekt
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Connect to Arkitekt
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 2,
                      mb: 2,
                    }}
                  >
                    <Typography variant="body1">Arkitekt Status:</Typography>
                    {arkitektConnected ? (
                      <>
                        <CheckCircleIcon style={{ color: green[500] }} />
                        <Typography color="success.main">Connected</Typography>
                      </>
                    ) : (
                      <>
                        <CancelIcon style={{ color: red[500] }} />
                        <Typography color="error.main">Disconnected</Typography>
                      </>
                    )}
                  </Box>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    onClick={connectToArkitekt}
                    disabled={arkitektConnected}
                    color="primary"
                    fullWidth
                  >
                    {arkitektConnected
                      ? "Connected to Arkitekt"
                      : "Connect to Arkitekt"}
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="textSecondary">
                    Arkitekt provides distributed processing capabilities for
                    STORM reconstruction with advanced algorithms and cloud
                    resources.
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      case 1: // Select Workflow
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Select Workflow
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Available Workflows</InputLabel>
                    <Select
                      value={selectedWorkflow}
                      onChange={(e) => setSelectedWorkflow(e.target.value)}
                      label="Available Workflows"
                      disabled={!arkitektConnected}
                    >
                      {availableWorkflows.map((workflow) => (
                        <MenuItem key={workflow.id} value={workflow.id}>
                          {workflow.name} - {workflow.description}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="outlined"
                    onClick={loadAvailableWorkflows}
                    disabled={!arkitektConnected}
                    fullWidth
                  >
                    Refresh Workflows
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="textSecondary">
                    Select a workflow that matches your experimental
                    requirements. Different workflows offer various algorithms
                    and processing options.
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      case 2: // Configure Parameters
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Configure Parameters
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Session ID"
                    value={acquisitionParameters.session_id || ""}
                    onChange={(e) =>
                      setAcquisitionParameter("session_id", e.target.value)
                    }
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Max Frames (-1 for unlimited)"
                    type="number"
                    value={acquisitionParameters.max_frames}
                    onChange={(e) =>
                      setAcquisitionParameter(
                        "max_frames",
                        parseInt(e.target.value) || -1
                      )
                    }
                    size="small"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Priority</InputLabel>
                    <Select
                      value={acquisitionParameters.priority || "normal"}
                      onChange={(e) =>
                        setAcquisitionParameter("priority", e.target.value)
                      }
                      label="Priority"
                    >
                      <MenuItem value="low">Low</MenuItem>
                      <MenuItem value="normal">Normal</MenuItem>
                      <MenuItem value="high">High</MenuItem>
                      <MenuItem value="urgent">Urgent</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={
                          acquisitionParameters.process_arkitekt || false
                        }
                        onChange={(e) =>
                          setAcquisitionParameter(
                            "process_arkitekt",
                            e.target.checked
                          )
                        }
                      />
                    }
                    label="Process with Arkitekt"
                  />
                </Grid>
                <Grid item xs={6}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={acquisitionParameters.save_results || false}
                        onChange={(e) =>
                          setAcquisitionParameter(
                            "save_results",
                            e.target.checked
                          )
                        }
                      />
                    }
                    label="Save Results"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      case 3: // Start Processing
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Start Processing
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 2,
                      mb: 2,
                    }}
                  >
                    <Typography variant="body1">Workflow Status:</Typography>
                    {workflowStatus === "running" ? (
                      <>
                        <CheckCircleIcon style={{ color: green[500] }} />
                        <Typography color="success.main">Running</Typography>
                      </>
                    ) : workflowStatus === "completed" ? (
                      <>
                        <CheckCircleIcon style={{ color: green[500] }} />
                        <Typography color="success.main">Completed</Typography>
                      </>
                    ) : (
                      <>
                        <CancelIcon style={{ color: red[500] }} />
                        <Typography color="error.main">Idle</Typography>
                      </>
                    )}
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant="contained"
                    onClick={startArkitektProcessing}
                    disabled={!selectedWorkflow || workflowStatus === "running"}
                    color="primary"
                    fullWidth
                  >
                    {workflowStatus === "running"
                      ? "Processing..."
                      : "Start Processing"}
                  </Button>
                </Grid>
                <Grid item xs={6}>
                  <Button
                    variant="contained"
                    onClick={stopArkitektProcessing}
                    disabled={workflowStatus !== "running"}
                    color="secondary"
                    fullWidth
                  >
                    Stop Processing
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" color="textSecondary">
                    Monitor the processing status in the next step. Results will
                    be displayed in real-time as they become available.
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      case 4: // Monitor Results
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Monitor Results
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="body1" gutterBottom>
                    Processing Status: {workflowStatus}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Results are displayed in the reconstruction panel on the
                    right. The workflow will continue processing until manually
                    stopped or completed.
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle1" gutterBottom>
                    Quick Actions
                  </Typography>
                  <Grid container spacing={1}>
                    <Grid item xs={6}>
                      <Button
                        variant="outlined"
                        size="small"
                        fullWidth
                        onClick={() => {
                          /* Export results */
                        }}
                      >
                        Export Results
                      </Button>
                    </Grid>
                    <Grid item xs={6}>
                      <Button
                        variant="outlined"
                        size="small"
                        fullWidth
                        onClick={() => {
                          /* Share workflow */
                        }}
                      >
                        Share Workflow
                      </Button>
                    </Grid>
                  </Grid>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      default:
        return "Unknown step";
    }
  };

  return (
    <Paper sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <Typography variant="h5" sx={{ p: 2, borderBottom: "1px solid #ddd" }}>
        STORM Arkitekt Controller
      </Typography>

      <Grid container sx={{ flex: 1, height: "calc(100vh - 80px)" }}>
        {/* Left Column - Live View */}
        <Grid
          item
          xs={4}
          sx={{ borderRight: "1px solid #ddd", height: "100%" }}
        >
          <Box
            sx={{
              p: 2,
              height: "100%",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Typography variant="h6" gutterBottom>
              Live View Stream
            </Typography>
            <Box
              sx={{ flex: 1, minHeight: 400, border: "1px solid #ccc", mb: 2 }}
            >
              <LiveViewControlWrapper />
            </Box>
            <Typography variant="h6" gutterBottom>
              Live View Settings
            </Typography>
            <LiveViewSettings />
          </Box>
        </Grid>

        {/* Middle Column - Arkitekt Settings Flow */}
        <Grid
          item
          xs={4}
          sx={{ borderRight: "1px solid #ddd", height: "100%" }}
        >
          <Box sx={{ p: 2, height: "100%", overflow: "auto" }}>
            <Typography variant="h6" gutterBottom>
              Arkitekt Settings
            </Typography>

            <Stepper activeStep={activeStep} orientation="vertical">
              {steps.map((label, index) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                  <StepContent>
                    {renderStepContent(index)}
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="contained"
                        onClick={handleNext}
                        sx={{ mr: 1 }}
                        disabled={index === steps.length - 1}
                      >
                        {index === steps.length - 1 ? "Finish" : "Continue"}
                      </Button>
                      <Button
                        disabled={index === 0}
                        onClick={handleBack}
                        sx={{ mr: 1 }}
                      >
                        Back
                      </Button>
                    </Box>
                  </StepContent>
                </Step>
              ))}
            </Stepper>
          </Box>
        </Grid>

        {/* Right Column - Live Reconstruction */}
        <Grid item xs={4} sx={{ height: "100%" }}>
          <Box
            sx={{
              p: 2,
              height: "100%",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Tabs
              value={reconstructionTabIndex}
              onChange={handleReconstructionTabChange}
              aria-label="reconstruction tabs"
            >
              <Tab label="Live Reconstruction" />
              <Tab label="Statistics" />
            </Tabs>

            {/* Live Reconstruction Tab */}
            <TabPanel value={reconstructionTabIndex} index={0}>
              <Tabs
                value={visualizationTabIndex}
                onChange={handleVisualizationTabChange}
                aria-label="visualization tabs"
                variant="fullWidth"
              >
                <Tab label="Rendering" />
                <Tab label="XY Plot" />
                <Tab label="Brightfield" />
              </Tabs>

              <TabPanel value={visualizationTabIndex} index={0}>
                <Box sx={{ textAlign: "center" }}>
                  <Typography variant="h6" gutterBottom>
                    Arkitekt Reconstruction
                  </Typography>
                  {reconImage ? (
                    <img
                      src={reconImage}
                      alt="Arkitekt STORM Reconstruction"
                      style={{ maxWidth: "100%", maxHeight: 400 }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: "100%",
                        height: 300,
                        backgroundColor: "#f5f5f5",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        border: "1px dashed #ccc",
                      }}
                    >
                      <Typography color="textSecondary">
                        No reconstruction available
                      </Typography>
                    </Box>
                  )}
                </Box>
              </TabPanel>

              <TabPanel value={visualizationTabIndex} index={1}>
                <Typography variant="h6" gutterBottom>
                  XY Plot
                </Typography>
                <STORMPlot data={plotData} title="Arkitekt Localizations" />
              </TabPanel>

              <TabPanel value={visualizationTabIndex} index={2}>
                <Box sx={{ textAlign: "center" }}>
                  <Typography variant="h6" gutterBottom>
                    Brightfield
                  </Typography>
                  <Box
                    sx={{
                      width: "100%",
                      height: 300,
                      backgroundColor: "#f5f5f5",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px dashed #ccc",
                    }}
                  >
                    <Typography color="textSecondary">
                      Brightfield imaging via Arkitekt workflow
                    </Typography>
                  </Box>
                </Box>
              </TabPanel>
            </TabPanel>

            {/* Statistics Tab */}
            <TabPanel value={reconstructionTabIndex} index={1}>
              <Typography variant="h6" gutterBottom>
                Arkitekt Statistics
              </Typography>
              {statistics ? (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Total Localizations: {statistics.totalLocalizations || 0}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Processing Rate: {statistics.processingRate || 0} fps
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Workflow: {statistics.workflow || "N/A"}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Quality Score: {statistics.qualityScore || 0}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="body2">
                      Cloud Resources:{" "}
                      {statistics.cloudResources || "Local processing"}
                    </Typography>
                  </Grid>
                </Grid>
              ) : (
                <Typography color="textSecondary">
                  No statistics available
                </Typography>
              )}
            </TabPanel>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default STORMControllerArkitekt;
