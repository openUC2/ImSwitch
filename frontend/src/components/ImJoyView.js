import React, { useEffect, useState, useRef, useCallback } from "react";
import { Box, Typography, Button, CircularProgress } from "@mui/material";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

const ImJoyView = ({ sharedImage }) => {
  // Get connection settings from Redux
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState,
  );
  const [imjoyAPI, setImjoyAPI] = useState(null);
  const [loading, setLoading] = useState(true);
  // Track the last shared image we processed to avoid re-opening the same one
  const processedImageRef = useRef(null);

  useEffect(() => {
    let cancelled = false;

    async function initImJoy() {
      // If the loader is already available (script was cached), skip adding it
      if (!window.loadImJoyBasicApp) {
        await new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = "https://lib.imjoy.io/imjoy-loader.js";
          script.async = true;
          script.onload = resolve;
          script.onerror = reject;
          document.body.appendChild(script);
        });
      }
      if (cancelled) return;

      const container = document.getElementById("imjoy-window-container");
      if (!container) return;

      try {
        const app = await window.loadImJoyBasicApp({
          process_url_query: false,
          show_window_title: false,
          show_progress_bar: true,
          show_empty_window: true,
          menu_style: { position: "absolute", right: 0, top: "2px" },
          window_style: { width: "100%", height: "100%" },
          main_container: null,
          menu_container: "imjoy-menu-container",
          window_manager_container: "imjoy-window-container",
        });
        if (cancelled) return;
        setImjoyAPI(app.imjoy.api);
        setLoading(false);
        app.addMenuItem({
          label: "➕ Load Plugin",
          callback() {
            const uri = prompt("Please type an ImJoy plugin URL");
            if (uri) app.loadPlugin(uri);
          },
        });
      } catch (err) {
        console.error("Failed to load ImJoy app:", err);
        if (!cancelled) setLoading(false);
      }
    }

    initImJoy();
    return () => {
      cancelled = true;
    };
  }, []);

  // Open shared image when both API and image data are ready
  const openSharedImage = useCallback(async (api, image) => {
    if (!api || !image) return;
    try {
      const response = await fetch(image.url);
      if (!response.ok) {
        console.error("Failed to fetch shared image:", response.statusText);
        return;
      }
      const arrayBuffer = await response.arrayBuffer();
      let ij = await api.getWindow("ImageJ.JS");
      if (!ij) {
        ij = await api.createWindow({
          src: "https://ij.imjoy.io",
          name: "ImageJ.JS",
          fullscreen: true,
        });
      } else {
        await ij.show();
      }
      await ij.viewImage(arrayBuffer, { name: image.name });
    } catch (error) {
      console.error("Error opening shared image:", error);
    }
  }, []);

  useEffect(() => {
    // Only process if we have a new image that we haven't handled yet
    if (
      sharedImage &&
      imjoyAPI &&
      processedImageRef.current !== sharedImage.url
    ) {
      processedImageRef.current = sharedImage.url;
      openSharedImage(imjoyAPI, sharedImage);
    }
  }, [sharedImage, imjoyAPI, openSharedImage]);

  const handleSnapAndSend = async () => {
    if (!imjoyAPI) return;
    try {
      const response = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/RecordingController/snapNumpyToFastAPI?resizeFactor=1`,
      );
      const arrayBuffer = await response.arrayBuffer();
      let ij = await imjoyAPI.getWindow("ImageJ.JS");
      if (!ij) {
        ij = await imjoyAPI.createWindow({
          src: "https://ij.imjoy.io",
          name: "ImageJ.JS",
          fullscreen: true,
        });
      } else {
        await ij.show();
      }
      await ij.viewImage(arrayBuffer, { name: "snapped-image.jpeg" });
    } catch (error) {
      console.error("Error snapping/sending:", error);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        position: "relative",
        height: "100%",
        width: "100%",
        p: 2,
      }}
    >
      {/* Header section - fixed height */}
      <Box
        sx={{ display: "flex", flexDirection: "column", gap: 1, flexShrink: 0 }}
      >
        <Typography variant="h6">ImJoy Integration Page</Typography>
        <Button
          variant="contained"
          onClick={handleSnapAndSend}
          disabled={!imjoyAPI}
          sx={{ alignSelf: "flex-start" }}
        >
          Snap and Send to ImJoy
        </Button>
        {loading && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <CircularProgress size={20} />
            <Typography variant="body2">Loading ImJoy...</Typography>
          </Box>
        )}
      </Box>

      {/* Menu container - fixed height */}
      <Box
        id="imjoy-menu-container"
        sx={{
          width: "100%",
          height: 40,
          border: "1px solid #ccc",
          flexShrink: 0,
        }}
      ></Box>

      {/* Window container - grows to fill remaining space */}
      <Box
        id="imjoy-window-container"
        sx={{
          width: "100%",
          flex: 1,
          minHeight: 400,
          border: "1px solid #ccc",
          overflow: "hidden",
        }}
      ></Box>
    </Box>
  );
};

export default ImJoyView;
