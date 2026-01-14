// src/components/MazeGameController.js
import React, { useEffect, useState, useRef, useCallback } from "react";
import { useDispatch, useSelector } from "react-redux";
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
  Divider,
  Slider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  Chip,
} from "@mui/material";
import { green, red, blue, grey } from "@mui/material/colors";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { Line } from "react-chartjs-2";
import { useTheme } from "@mui/material/styles";
import * as mazeGameSlice from "../state/slices/MazeGameSlice.js";
import * as positionSlice from "../state/slices/PositionSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import * as objectiveSlice from "../state/slices/ObjectiveSlice.js";
import LiveViewComponent from "../axon/LiveViewComponent.js";
import LiveViewerGL from "./LiveViewerGL.js";
import apiMazeGameControllerStartGame from "../backendapi/apiMazeGameControllerStartGame.js";
import apiMazeGameControllerStopGame from "../backendapi/apiMazeGameControllerStopGame.js";
import apiMazeGameControllerResetGame from "../backendapi/apiMazeGameControllerResetGame.js";
import apiMazeGameControllerGetState from "../backendapi/apiMazeGameControllerGetState.js";
import apiMazeGameControllerMoveToStartPosition from "../backendapi/apiMazeGameControllerMoveToStartPosition.js";
import apiMazeGameControllerSetCropSize from "../backendapi/apiMazeGameControllerSetCropSize.js";
import apiMazeGameControllerSetJumpThresholds from "../backendapi/apiMazeGameControllerSetJumpThresholds.js";
import apiMazeGameControllerSetHistory from "../backendapi/apiMazeGameControllerSetHistory.js";
import apiMazeGameControllerSetDownscale from "../backendapi/apiMazeGameControllerSetDownscale.js";
import apiMazeGameControllerSetPollInterval from "../backendapi/apiMazeGameControllerSetPollInterval.js";
import apiPositionerControllerMovePositioner from "../backendapi/apiPositionerControllerMovePositioner.js";

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`mazegame-tabpanel-${index}`}
      aria-labelledby={`mazegame-tab-${index}`}
      {...other}
    >
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
};

const CountdownOverlay = ({ countdown }) => {
  if (countdown === null) return null;

  return (
    <Box
      sx={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        zIndex: 10,
      }}
    >
      <Typography
        variant="h1"
        sx={{
          color: countdown === "GO!" ? green[500] : "white",
          fontSize: "120px",
          fontWeight: "bold",
          animation: "pulse 0.5s ease-in-out",
          "@keyframes pulse": {
            "0%": { transform: "scale(0.8)", opacity: 0 },
            "50%": { transform: "scale(1.2)", opacity: 1 },
            "100%": { transform: "scale(1)", opacity: 1 },
          },
        }}
      >
        {countdown}
      </Typography>
    </Box>
  );
};

const MazeGameController = ({ title = "Maze Game" }) => {
  const dispatch = useDispatch();
  const theme = useTheme();
  const mazeGameState = useSelector(mazeGameSlice.getMazeGameState);
  const positionState = useSelector(positionSlice.getPositionState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const objectiveState = useSelector(objectiveSlice.getObjectiveState);

  // Destructure state
  const {
    playerName,
    cropSize,
    jumpLow,
    jumpHigh,
    history,
    downscale,
    pollInterval,
    stepSize,
    startPosition,
    running,
    counter,
    elapsed,
    smoothMean,
    xyTrace,
    hallOfFame,
  } = mazeGameState;

  // Get current position from Redux
  const currentPosition = { x: positionState.x || 0, y: positionState.y || 0 };

  // Local state
  const [tabIndex, setTabIndex] = useState(0);
  const [countdown, setCountdown] = useState(null);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [nameInputValue, setNameInputValue] = useState("");
  const [wallHit, setWallHit] = useState(false);
  const [hudData, setHudData] = useState({
    stats: { fps: 0, bps: 0 },
    featureSupport: { webgl2: false, lz4: false },
    isWebGL: false,
    imageSize: { width: 0, height: 0 },
    viewTransform: { scale: 1, translateX: 0, translateY: 0 }
  });
  const previousCounterRef = useRef(counter);

  // Determine if we should use WebGL based on backend capabilities
  const useWebGL = liveStreamState.backendCapabilities.webglSupported && !liveStreamState.isLegacyBackend;

  // Handle HUD data updates from LiveViewerGL
  const handleHudDataUpdate = useCallback((data) => {
    setHudData(prevData => {
      if (!prevData) return data;
      
      const hasChanged = 
        prevData.stats?.fps !== data.stats?.fps ||
        prevData.stats?.bps !== data.stats?.bps ||
        prevData.imageSize?.width !== data.imageSize?.width ||
        prevData.imageSize?.height !== data.imageSize?.height ||
        prevData.viewTransform?.scale !== data.viewTransform?.scale ||
        prevData.viewTransform?.translateX !== data.viewTransform?.translateX ||
        prevData.viewTransform?.translateY !== data.viewTransform?.translateY ||
        prevData.isWebGL !== data.isWebGL ||
        JSON.stringify(prevData.featureSupport) !== JSON.stringify(data.featureSupport);
      
      return hasChanged ? data : prevData;
    });
  }, []);

  // Handle double-click for stage movement (optional for maze game)
  const handleImageDoubleClick = async (pixelX, pixelY, imageWidth, imageHeight) => {
    if (running) return; // Disable double-click during game
    
    try {
      const fovX = objectiveState.fovX || 1000;
      const fovY = objectiveState.fovY || (fovX * imageHeight / imageWidth);
      
      const centerX = imageWidth / 2;
      const centerY = imageHeight / 2;
      
      const relativeX = (pixelX - centerX) / imageWidth;
      const relativeY = (pixelY - centerY) / imageHeight;
      
      const moveX = relativeX * fovX;
      const moveY = relativeY * fovY;
      
      await apiPositionerControllerMovePositioner({
        axis: "X",
        dist: moveX,
        isAbsolute: false,
        isBlocking: false
      });
      
      await apiPositionerControllerMovePositioner({
        axis: "Y", 
        dist: -moveY,
        isAbsolute: false,
        isBlocking: false
      });
    } catch (error) {
      console.error("Failed to move stage:", error);
    }
  };

  // Detect wall hits (counter increment) and trigger red flash
  useEffect(() => {
    if (counter > previousCounterRef.current && running) {
      setWallHit(true);
      const timer = setTimeout(() => setWallHit(false), 2000);
      return () => clearTimeout(timer);
    }
    previousCounterRef.current = counter;
  }, [counter, running]);

  // Load hall of fame from localStorage on mount
  useEffect(() => {
    const savedHallOfFame = localStorage.getItem("mazeGameHallOfFame");
    if (savedHallOfFame) {
      try {
        const parsed = JSON.parse(savedHallOfFame);
        dispatch(mazeGameSlice.setHallOfFame(parsed));
      } catch (error) {
        console.error("Error loading hall of fame:", error);
      }
    }
  }, [dispatch]);

  // Save hall of fame to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("mazeGameHallOfFame", JSON.stringify(hallOfFame));
  }, [hallOfFame]);

  // Fetch initial game state
  useEffect(() => {
    const fetchInitialState = async () => {
      try {
        const state = await apiMazeGameControllerGetState();
        dispatch(mazeGameSlice.setGameState(state));
      } catch (error) {
        console.error("Error fetching initial game state:", error);
      }
    };

    fetchInitialState();
  }, [dispatch]);

  // Track position changes and add to trace when game is running
  useEffect(() => {
    if (running && (currentPosition.x !== 0 || currentPosition.y !== 0)) {
      // Add current position to trace
      dispatch(
        mazeGameSlice.addTracePoint({
          x: currentPosition.x,
          y: currentPosition.y,
          timestamp: Date.now(),
        })
      );
    }
  }, [running, currentPosition.x, currentPosition.y, dispatch]);

  // Handle countdown animation
  const startCountdown = () => {
    return new Promise((resolve) => {
      setCountdown(3);
      setTimeout(() => {
        setCountdown(2);
        setTimeout(() => {
          setCountdown(1);
          setTimeout(() => {
            setCountdown("GO!");
            setTimeout(() => {
              setCountdown(null);
              resolve();
            }, 500);
          }, 1000);
        }, 1000);
      }, 1000);
    });
  };

  // Game control handlers
  const handleStartGame = async () => {
    if (!playerName.trim()) {
      alert("Please enter your name before starting!");
      return;
    }

    try {
      // Clear previous trace
      dispatch(mazeGameSlice.clearTrace());

      // set min/max value for levels (use destructured values from state)
      await apiMazeGameControllerSetJumpThresholds(jumpLow, jumpHigh);

      // Start countdown
      await startCountdown();

      // Start the game
      await apiMazeGameControllerStartGame(startPosition.x, startPosition.y);
      dispatch(mazeGameSlice.setRunning(true));
    } catch (error) {
      console.error("Error starting game:", error);
      alert("Failed to start game. Check console for details.");
    }
  };

  const handleStopGame = async () => {
    try {
      await apiMazeGameControllerStopGame();
      dispatch(mazeGameSlice.setRunning(false));

      // Save to hall of fame
      if (playerName.trim() && elapsed > 0) {
        const result = {
          playerName: playerName.trim(),
          time: elapsed,
          counter,
          trace: [...xyTrace],
          timestamp: Date.now(),
        };
        dispatch(mazeGameSlice.addToHallOfFame(result));
      }
    } catch (error) {
      console.error("Error stopping game:", error);
    }
  };

  const handleResetGame = async () => {
    try {
      await apiMazeGameControllerResetGame();
      dispatch(mazeGameSlice.setRunning(false));
      dispatch(mazeGameSlice.setCounter(0));
      dispatch(mazeGameSlice.setElapsed(0));
      dispatch(mazeGameSlice.setSmoothMean(0));
      dispatch(mazeGameSlice.clearTrace());
    } catch (error) {
      console.error("Error resetting game:", error);
    }
  };

  const handleMoveToStart = async () => {
    try {
      await apiMazeGameControllerMoveToStartPosition(
        startPosition.x,
        startPosition.y
      );
    } catch (error) {
      console.error("Error moving to start:", error);
    }
  };

  // Parameter update handlers
  const handleCropSizeChange = async (value) => {
    dispatch(mazeGameSlice.setCropSize(value));
    try {
      await apiMazeGameControllerSetCropSize(value);
    } catch (error) {
      console.error("Error setting crop size:", error);
    }
  };

  const handleJumpThresholdsChange = async (low, high) => {
    dispatch(mazeGameSlice.setJumpLow(low));
    dispatch(mazeGameSlice.setJumpHigh(high));
    try {
      await apiMazeGameControllerSetJumpThresholds(low, high);
    } catch (error) {
      console.error("Error setting jump thresholds:", error);
    }
  };

  const handleHistoryChange = async (value) => {
    dispatch(mazeGameSlice.setHistory(value));
    try {
      await apiMazeGameControllerSetHistory(value);
    } catch (error) {
      console.error("Error setting history:", error);
    }
  };

  const handleDownscaleChange = async (value) => {
    dispatch(mazeGameSlice.setDownscale(value));
    try {
      await apiMazeGameControllerSetDownscale(value);
    } catch (error) {
      console.error("Error setting downscale:", error);
    }
  };

  const handlePollIntervalChange = async (value) => {
    dispatch(mazeGameSlice.setPollInterval(value));
    try {
      await apiMazeGameControllerSetPollInterval(value);
    } catch (error) {
      console.error("Error setting poll interval:", error);
    }
  };

  // Joystick control
  const handleJoystickMove = async (axis, distance) => {
    try {
      await apiPositionerControllerMovePositioner({
        axis,
        dist: (distance * stepSize) / 100, // Use stepSize from slice, scaled by 100
        isAbsolute: false,
        isBlocking: false,
      });
    } catch (error) {
      console.error("Error moving stage:", error);
    }
  };

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  // Trajectory chart data
  const trajectoryChartData = {
    datasets: [
      {
        label: "Player Path",
        data: xyTrace.map((point) => ({ x: point.x, y: point.y })),
        borderColor: blue[500],
        backgroundColor: blue[200],
        showLine: true,
        pointRadius: 2,
      },
    ],
  };

  const trajectoryChartOptions = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 1,
    scales: {
      x: {
        type: "linear",
        title: {
          display: true,
          text: "X Position",
        },
      },
      y: {
        type: "linear",
        title: {
          display: true,
          text: "Y Position",
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
    },
  };

  return (
    <Paper
      elevation={3}
      sx={{ width: "100%", height: "100vh", overflow: "auto" }}
    >
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs
          value={tabIndex}
          onChange={(e, newValue) => setTabIndex(newValue)}
        >
          <Tab label="Game" />
          <Tab
            label="Hall of Fame"
            icon={<EmojiEventsIcon />}
            iconPosition="start"
          />
        </Tabs>
      </Box>

      {/* Game Tab */}
      <TabPanel value={tabIndex} index={0}>
        <Grid container spacing={2}>
          {/* Left Panel - Game View */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  {title}
                </Typography>
                <Divider sx={{ my: 2 }} />

                {/* Player Name Input */}
                {!playerName && (
                  <Box sx={{ mb: 3 }}>
                    <TextField
                      fullWidth
                      label="Enter Your Name"
                      value={nameInputValue}
                      onChange={(e) => setNameInputValue(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === "Enter" && nameInputValue.trim()) {
                          dispatch(
                            mazeGameSlice.setPlayerName(nameInputValue.trim())
                          );
                        }
                      }}
                      onBlur={() => {
                        if (nameInputValue.trim()) {
                          dispatch(
                            mazeGameSlice.setPlayerName(nameInputValue.trim())
                          );
                        }
                      }}
                      disabled={running}
                      placeholder="Press Enter to confirm"
                    />
                  </Box>
                )}

                {playerName && (
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={`Player: ${playerName}`}
                      color="primary"
                      sx={{ mr: 2 }}
                    />
                    {!running && (
                      <Button
                        size="small"
                        onClick={() =>
                          dispatch(mazeGameSlice.setPlayerName(""))
                        }
                      >
                        Change Name
                      </Button>
                    )}
                  </Box>
                )}

                {/* Game Controls */}
                <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
                  <Button
                    variant="contained"
                    color="success"
                    startIcon={<PlayArrowIcon />}
                    onClick={handleStartGame}
                    disabled={running || !playerName}
                  >
                    Start Game
                  </Button>
                  <Button
                    variant="contained"
                    color="error"
                    startIcon={<StopIcon />}
                    onClick={handleStopGame}
                    disabled={!running}
                  >
                    Stop Game
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<RestartAltIcon />}
                    onClick={handleResetGame}
                    disabled={running}
                  >
                    Reset
                  </Button>
                </Box>

                {/* Timer and Counter */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={4}>
                    <Card
                      sx={{
                        bgcolor:
                          theme.palette.mode === "dark" ? grey[800] : green[50],
                      }}
                    >
                      <CardContent>
                        <Typography variant="h6" color="textSecondary">
                          Timer
                        </Typography>
                        <Typography variant="h3">
                          {formatTime(elapsed)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={4}>
                    <Card
                      sx={{
                        bgcolor:
                          theme.palette.mode === "dark" ? grey[800] : red[50],
                      }}
                    >
                      <CardContent>
                        <Typography variant="h6" color="textSecondary">
                          Wall Hits
                        </Typography>
                        <Typography variant="h3">{counter}</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={4}>
                    <Card
                      sx={{
                        bgcolor:
                          theme.palette.mode === "dark" ? grey[800] : blue[50],
                      }}
                    >
                      <CardContent>
                        <Typography variant="h6" color="textSecondary">
                          Smooth Mean
                        </Typography>
                        <Typography variant="h3">
                          {(smoothMean ?? 0).toFixed(2)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {/* Live Preview with crop box overlay */}
                <Box sx={{ position: "relative", mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Live Preview
                  </Typography>
                  <Box
                    sx={{
                      width: "100%",
                      border: "2px solid #ccc",
                      borderRadius: 1,
                      overflow: "hidden",
                      position: "relative",
                      bgcolor: "#000",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    {/* Live stream component */}
                    {useWebGL ? (
                      <LiveViewerGL 
                        onDoubleClick={handleImageDoubleClick}
                        onImageLoad={(width, height) => {
                          // Optional: handle image load events
                        }}
                        onHudDataUpdate={handleHudDataUpdate}
                      />
                    ) : (
                      <LiveViewComponent 
                        useFastMode={true} 
                        onDoubleClick={handleImageDoubleClick}
                      />
                    )}
                    
                    {/* Crop size rectangle overlay */}
                    {hudData.imageSize.width > 0 && hudData.imageSize.height > 0 && (
                      <Box
                        sx={{
                          position: "absolute",
                          top: "50%",
                          left: "50%",
                          transform: "translate(-50%, -50%)",
                          width: `${cropSize}px`,
                          height: `${cropSize}px`,
                          border: wallHit ? `4px solid ${red[500]}` : `2px solid ${green[500]}`,
                          borderRadius: 1,
                          pointerEvents: "none",
                          transition: "border 0.3s ease",
                          boxShadow: wallHit ? `0 0 20px ${red[500]}` : "none",
                          zIndex: 1002,
                        }}
                      />
                    )}
                    
                    <CountdownOverlay countdown={countdown} />
                  </Box>
                </Box>

                {/* Trajectory + Joystick side-by-side (responsive) */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={12} md={8}>
                    <Typography variant="h6" gutterBottom>
                      Your Path
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      {xyTrace.length > 0 ? (
                        <Line
                          data={trajectoryChartData}
                          options={trajectoryChartOptions}
                        />
                      ) : (
                        <Box
                          sx={{
                            height: "100%",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            border: "1px solid #ccc",
                            borderRadius: 1,
                          }}
                        >
                          <Typography color="textSecondary">
                            Path will be displayed here during the game
                          </Typography>
                        </Box>
                      )}
                    </Box>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Typography variant="h6" gutterBottom>
                      Joystick Control
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: "repeat(3, 80px)",
                        gap: 1,
                        justifyContent: "center",
                      }}
                    >
                      <Box />
                      <Button
                        variant="contained"
                        onClick={() => handleJoystickMove("Y", 100)}
                      >
                        ↑
                      </Button>
                      <Box />
                      <Button
                        variant="contained"
                        onClick={() => handleJoystickMove("X", 100)}
                      >
                        ←
                      </Button>
                      <Box />
                      <Button
                        variant="contained"
                        onClick={() => handleJoystickMove("X", -100)}
                      >
                        →
                      </Button>
                      <Box />
                      <Button
                        variant="contained"
                        onClick={() => handleJoystickMove("Y", -100)}
                      >
                        ↓
                      </Button>
                      <Box />
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          {/* Right Panel - Settings and Position */}
          <Grid item xs={12} md={4}>
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Current Position
                </Typography>
                <Typography>
                  X: {(currentPosition.x ?? 0).toFixed(2)}
                </Typography>
                <Typography>
                  Y: {(currentPosition.y ?? 0).toFixed(2)}
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Start Position
                </Typography>
                <TextField
                  fullWidth
                  label="X"
                  type="number"
                  value={startPosition.x}
                  onChange={(e) =>
                    dispatch(
                      mazeGameSlice.setStartPosition({
                        ...startPosition,
                        x: parseFloat(e.target.value) || 0,
                      })
                    )
                  }
                  disabled={running}
                  sx={{ mb: 2 }}
                />
                <TextField
                  fullWidth
                  label="Y"
                  type="number"
                  value={startPosition.y}
                  onChange={(e) =>
                    dispatch(
                      mazeGameSlice.setStartPosition({
                        ...startPosition,
                        y: parseFloat(e.target.value) || 0,
                      })
                    )
                  }
                  disabled={running}
                  sx={{ mb: 2 }}
                />
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={handleMoveToStart}
                  disabled={running}
                >
                  Move to Start
                </Button>
              </CardContent>
            </Card>

            {playerName && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Game Parameters
                  </Typography>

                  <Typography gutterBottom>Crop Size: {cropSize}</Typography>
                  <Slider
                    value={cropSize}
                    onChange={(e, val) => handleCropSizeChange(val)}
                    min={1}
                    max={200}
                    disabled={running}
                    sx={{ mb: 3 }}
                  />

                  <Typography gutterBottom>
                    Jump Thresholds: {jumpLow} - {jumpHigh}
                  </Typography>
                  <Slider
                    value={[jumpLow, jumpHigh]}
                    onChange={(e, val) =>
                      handleJumpThresholdsChange(val[0], val[1])
                    }
                    min={0}
                    max={1000}
                    step={0.05}
                    disabled={running}
                    sx={{ mb: 3 }}
                  />

                  <Typography gutterBottom>History: {history}</Typography>
                  <Slider
                    value={history}
                    onChange={(e, val) => handleHistoryChange(val)}
                    min={1}
                    max={10}
                    disabled={running}
                    sx={{ mb: 3 }}
                  />

                  <Typography gutterBottom>Downscale: {downscale}</Typography>
                  <Slider
                    value={downscale}
                    onChange={(e, val) => handleDownscaleChange(val)}
                    min={1}
                    max={4}
                    disabled={running}
                    sx={{ mb: 3 }}
                  />

                  <Typography gutterBottom>Step Size: {stepSize}</Typography>
                  <Slider
                    value={stepSize}
                    onChange={(e, val) =>
                      dispatch(mazeGameSlice.setStepSize(val))
                    }
                    min={10}
                    max={500}
                    step={10}
                  />
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      </TabPanel>

      {/* Hall of Fame Tab */}
      <TabPanel value={tabIndex} index={1}>
        <Typography variant="h5" gutterBottom>
          🏆 Hall of Fame
        </Typography>
        <Divider sx={{ my: 2 }} />

        {hallOfFame.length === 0 ? (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: 300,
            }}
          >
            <Typography variant="h6" color="textSecondary">
              No records yet. Be the first to complete the maze!
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rank</TableCell>
                  <TableCell>Player</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Wall Hits</TableCell>
                  <TableCell>Date</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {hallOfFame.map((entry, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      {index === 0 && "🥇"}
                      {index === 1 && "🥈"}
                      {index === 2 && "🥉"}
                      {index > 2 && index + 1}
                    </TableCell>
                    <TableCell>{entry.playerName}</TableCell>
                    <TableCell>{formatTime(entry.time)}</TableCell>
                    <TableCell>{entry.counter}</TableCell>
                    <TableCell>
                      {new Date(entry.timestamp).toLocaleDateString()}
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        onClick={() => setSelectedPlayer(entry)}
                      >
                        View Path
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {/* Path View Dialog */}
        <Dialog
          open={selectedPlayer !== null}
          onClose={() => setSelectedPlayer(null)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>{selectedPlayer?.playerName}'s Path</DialogTitle>
          <DialogContent>
            {selectedPlayer && (
              <Box sx={{ height: 400, mt: 2 }}>
                <Line
                  data={{
                    datasets: [
                      {
                        label: "Path",
                        data: selectedPlayer.trace.map((p) => ({
                          x: p.x,
                          y: p.y,
                        })),
                        borderColor: blue[500],
                        backgroundColor: blue[200],
                        showLine: true,
                        pointRadius: 2,
                      },
                    ],
                  }}
                  options={trajectoryChartOptions}
                />
              </Box>
            )}
          </DialogContent>
        </Dialog>
      </TabPanel>
    </Paper>
  );
};

export default MazeGameController;
