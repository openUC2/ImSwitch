# Maze Game Controller

## Overview
The Maze Game Controller is an interactive game interface that transforms the ImSwitch microscope stage into a playable maze challenge. Players navigate through a maze displayed on the live video stream using joystick controls.

## Features

### 🎮 Core Gameplay
- **Joystick Control**: Navigate the microscope stage using arrow buttons
- **Live Preview**: View the processed maze frame updated every 250ms
- **Real-time Position**: Track your X/Y position as you move through the maze
- **Move to Start**: Button to return to the starting position

### 👤 Player Experience
- **Player Name**: Enter your name before starting the game
- **Game Parameters**: Customize maze detection parameters (crop size, thresholds, etc.)
- **Countdown Animation**: 3-2-1-GO countdown before game starts

### 🎯 Game Metrics
- **Timer**: Elapsed time displayed in MM:SS format
- **Wall Hits Counter**: Tracks how many times you cross maze walls
- **XY Trajectory**: Live plot showing your path through the maze

### 🏆 Hall of Fame
- **Leaderboard**: Sorted by best completion time
- **Player Records**: View all past game sessions
- **Path Replay**: Visualize any player's trajectory
- **Persistent Storage**: Records saved in localStorage

## Usage

### Starting a Game
1. Navigate to "Maze Game" in the Apps menu
2. Enter your player name
3. (Optional) Adjust game parameters in the right panel
4. Set your start position or use "Move to Start"
5. Click "Start Game"
6. Navigate through the maze using the joystick
7. Click "Stop Game" when finished (or it will auto-stop when you reach the goal)

### Game Parameters
- **Crop Size** (50-200): Size of the detection window
- **Jump Thresholds** (5-100): Low and high thresholds for wall detection
- **History** (10-100): Number of frames to track
- **Downscale** (1-4): Image downscaling factor
- **Poll Interval** (100-1000ms): Preview update frequency

### Hall of Fame
- Switch to the "Hall of Fame" tab to view past results
- Click "View Path" to see any player's trajectory
- Results are automatically saved when you stop the game

## Backend Requirements

The frontend expects the following backend endpoints to be available:

### Game Control
- `GET /MazeGameController/startGame?startX=<x>&startY=<y>` - Start the game
- `GET /MazeGameController/stopGame` - Stop the game
- `GET /MazeGameController/resetGame` - Reset game state
- `GET /MazeGameController/getState` - Get current game state (running, counter, elapsed_s)

### Data Retrieval
- `GET /MazeGameController/getCounter` - Get wall hit counter
- `GET /MazeGameController/getElapsedSeconds` - Get elapsed time
- `GET /MazeGameController/getLatestProcessedPreview` - Get processed maze image (base64 JPEG)

### Position Control
- `GET /MazeGameController/moveToStartPosition?x=<x>&y=<y>` - Move stage to start position
- `GET /PositionerController/getPositions` - Get current XY position
- `GET /PositionerController/movePositioner` - Move stage (for joystick)

### Parameter Configuration
- `GET /MazeGameController/setCropSize?size=<size>`
- `GET /MazeGameController/setJumpThresholds?low=<low>&high=<high>`
- `GET /MazeGameController/setHistory?history=<history>`
- `GET /MazeGameController/setDownscale?downscale=<downscale>`
- `GET /MazeGameController/setPollInterval?interval=<interval>`

### WebSocket Signals
The component listens for the following socket.io signals for real-time updates:

- `sigGameState` - Game state updates (running, counter, elapsed_s)
- `sigCounterUpdated` - Counter increment events
- `sigPreviewUpdated` - New preview frame available (jpeg_b64)

## Redux State

The component uses the `mazeGameState` Redux slice which includes:

```javascript
{
  playerName: string,
  cropSize: number,
  jumpLow: number,
  jumpHigh: number,
  history: number,
  downscale: number,
  pollInterval: number,
  running: boolean,
  counter: number,
  elapsed: number,
  xyTrace: Array<{x, y, timestamp}>,
  hallOfFame: Array<{playerName, time, counter, trace, timestamp}>
}
```

The state is persisted across sessions (except `running`, `counter`, `elapsed`, and `xyTrace`).

## Technical Details

### Chart.js Integration
The component uses `react-chartjs-2` with `chart.js` for trajectory visualization:
- XY scatter plot with connected lines
- Real-time updates during gameplay
- Static plot in Hall of Fame

### localStorage
Hall of Fame data is stored in `localStorage` under the key `mazeGameHallOfFame`.
Data persists across browser sessions and is automatically loaded on component mount.

### Polling Strategy
- Preview images: Polled at the interval specified by `pollInterval` (default 250ms)
- Stage position: Polled every 100ms for smooth trajectory tracking
- Both polling loops stop when the game is not running

## Dependencies
- @mui/material - UI components
- react-chartjs-2 & chart.js - Trajectory plotting
- socket.io-client - WebSocket communication
- @reduxjs/toolkit - State management
- redux-persist - State persistence

## Future Enhancements
- [ ] Replay mode to watch saved trajectories
- [ ] Multiple maze difficulty levels
- [ ] Score calculation based on time + wall hits
- [ ] Multiplayer/race mode
- [ ] Export trajectory data as CSV/JSON
- [ ] Maze editor to create custom mazes
