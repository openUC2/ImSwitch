// src/state/slices/StageMapSlice.js
// Redux slice for the Stage Map (MicroMagellan-style live stitching) app.
// Tiles arrive via the sigStageMapTileAdded socket signal and are placed on
// a pan/zoom canvas in stage coordinates by StageMapController.jsx.

import { createSlice } from "@reduxjs/toolkit";

// Keep memory bounded: each preview tile is a ~20-50 KB base64 JPEG.
const MAX_TILES = 2000;

// Colors assigned to channels in order of first appearance.
// First channel renders as plain grayscale/white (brightfield-style).
export const CHANNEL_COLOR_PALETTE = [
  "#ffffff", // white (no tint)
  "#00e676", // green
  "#ff5252", // red
  "#448aff", // blue
  "#e040fb", // magenta
  "#ffd740", // yellow
  "#18ffff", // cyan
];

const initialState = {
  // [{id, x, y, widthUm, heightUm, channel, image}] - image is base64 jpeg
  tiles: [],
  // channel name -> {color, visible}
  channels: {},
  isMapping: false,
  status: {
    tileCount: 0,
    sessionPath: "",
    activeChannel: "",
    pixelSizeUm: 1,
    fovX: 0,
    fovY: 0,
    detectorName: "",
    lastError: "",
  },
  lastTileId: -1,
};

const ensureChannel = (state, channelName) => {
  if (!state.channels[channelName]) {
    const usedColors = Object.values(state.channels).map((c) => c.color);
    const color =
      CHANNEL_COLOR_PALETTE.find((c) => !usedColors.includes(c)) ||
      CHANNEL_COLOR_PALETTE[
        Object.keys(state.channels).length % CHANNEL_COLOR_PALETTE.length
      ];
    state.channels[channelName] = { color, visible: true };
  }
};

const stageMapSlice = createSlice({
  name: "stageMap",
  initialState,
  reducers: {
    addTile: (state, action) => {
      const tile = action.payload;
      if (!tile || tile.x === undefined || tile.y === undefined) return;
      // Replace an existing tile with the same id (e.g. on reload)
      const existingIndex = state.tiles.findIndex((t) => t.id === tile.id);
      if (existingIndex >= 0) {
        state.tiles[existingIndex] = tile;
      } else {
        state.tiles.push(tile);
        if (state.tiles.length > MAX_TILES) {
          state.tiles.splice(0, state.tiles.length - MAX_TILES);
        }
      }
      state.lastTileId = Math.max(state.lastTileId, tile.id ?? -1);
      ensureChannel(state, tile.channel || "default");
    },

    setTiles: (state, action) => {
      const tiles = Array.isArray(action.payload) ? action.payload : [];
      state.tiles = tiles.slice(-MAX_TILES);
      state.lastTileId = tiles.reduce((m, t) => Math.max(m, t.id ?? -1), -1);
      tiles.forEach((t) => ensureChannel(state, t.channel || "default"));
    },

    clearTiles: (state) => {
      state.tiles = [];
      state.channels = {};
      state.lastTileId = -1;
    },

    setChannelVisible: (state, action) => {
      const { channel, visible } = action.payload;
      ensureChannel(state, channel);
      state.channels[channel].visible = visible;
    },

    setChannelColor: (state, action) => {
      const { channel, color } = action.payload;
      ensureChannel(state, channel);
      state.channels[channel].color = color;
    },

    setIsMapping: (state, action) => {
      state.isMapping = Boolean(action.payload);
    },

    setStatus: (state, action) => {
      state.status = { ...state.status, ...(action.payload || {}) };
      if (action.payload?.isRunning !== undefined) {
        state.isMapping = Boolean(action.payload.isRunning);
      }
      // make channels reported by the backend selectable even before a tile arrived
      (action.payload?.channels || []).forEach((c) => ensureChannel(state, c));
    },
  },
});

export const {
  addTile,
  setTiles,
  clearTiles,
  setChannelVisible,
  setChannelColor,
  setIsMapping,
  setStatus,
} = stageMapSlice.actions;

export const getStageMapState = (state) => state.stageMapState;

export default stageMapSlice.reducer;
