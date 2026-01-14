import React, { useEffect, useState } from "react";
import { Box, Typography, Button } from "@mui/material";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

const ImJoyView = ({ sharedImage }) => {
  // Get connection settings from Redux
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState
  );
  const [imjoyAPI, setImjoyAPI] = useState(null);

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://lib.imjoy.io/imjoy-loader.js";
    script.async = true;
    script.onload = async () => {
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
      setImjoyAPI(app.imjoy.api);
      app.addMenuItem({
        label: "➕ Load Plugin",
        callback() {
          const uri = prompt("Please type an ImJoy plugin URL");
          if (uri) app.loadPlugin(uri);
        },
      });
    };
    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  useEffect(() => {
    if (sharedImage && imjoyAPI) {
      handleOpenSharedImage();
    }
  }, [sharedImage, imjoyAPI]);

  const handleSnapAndSend = async () => {
    if (!imjoyAPI) return;
    try {
      const response = await fetch(
        `${hostIP}:${hostPort}/imswitch/api/RecordingController/snapNumpyToFastAPI?resizeFactor=1`
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

  const handleOpenSharedImage = async () => {
    if (!imjoyAPI || !sharedImage) return;
    try {
      const response = await fetch(sharedImage.url);
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
      await ij.viewImage(arrayBuffer, { name: sharedImage.name });
    } catch (error) {
      console.error("Error opening shared image:", error);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 2,
        position: "relative",
      }}
    >
      <Typography variant="h6" gutterBottom>
        ImJoy Integration Page
      </Typography>
      <Button variant="contained" onClick={handleSnapAndSend}>
        Snap and Send to ImJoy
      </Button>
      <Box
        id="imjoy-menu-container"
        sx={{ width: "100%", height: 40, border: "1px solid #ccc", mb: 2 }}
      ></Box>
      <Box
        id="imjoy-window-container"
        sx={{ width: "100%", height: 400, border: "1px solid #ccc" }}
      ></Box>
    </Box>
  );
};

export default ImJoyView;
