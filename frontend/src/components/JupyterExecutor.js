import React, { useEffect, useState } from "react";
import {
  AppBar,
  Toolbar,
  Typography,
  TextField,
  Box,
  IconButton,
  Tooltip,
} from "@mui/material";
import { OpenInNew, Refresh, Edit } from "@mui/icons-material";
import { useDispatch, useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import { setNotification } from "../state/slices/NotificationSlice.js";

const JupyterExecutor = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const dispatch = useDispatch();
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  const [jupyterUrl, setJupyterUrl] = useState(null);
  const [editableUrl, setEditableUrl] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);

  useEffect(() => {
    const fetchNotebookUrl = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/jupyternotebookurl`,
        );
        const data = await response.json();
        const notebookUrl = data["url"]; // e.g. http://192.168.1.100:8888/jupyter/
        const serverUrl = new URL(notebookUrl);
        const jupyterPath = data["path"] || serverUrl.pathname; // e.g. /jupyter/
        const jupyterPort = data["port"] || serverUrl.port; // e.g. 8888

        // The server only knows its own address, which the browser may not be
        // able to reach, so rebuild the URL against the host we are already
        // talking to and take the first candidate that answers:
        //   1. proxied on the API port  - Docker/Caddy, e.g. http://192.168.178.76:80/jupyter/
        //   2. direct on the Jupyter port - local instance, e.g. http://100.100.43.118:8888/jupyter/
        //   3. whatever the server reported - browser running on the server itself
        const candidates = [
          `${hostIP}:${hostPort}${jupyterPath}`,
          `${hostIP}:${jupyterPort}${jupyterPath}`,
          notebookUrl,
        ];

        setJupyterUrl(candidates[0]);
        setEditableUrl(candidates[0]);

        const validateUrl = async (url) => {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 4000);
          try {
            const response = await fetch(url, {
              method: "HEAD",
              signal: controller.signal,
            });
            return response.ok;
          } catch (validationError) {
            return false;
          } finally {
            clearTimeout(timeoutId);
          }
        };

        for (const candidate of candidates) {
          if (!(await validateUrl(candidate))) continue;
          if (candidate !== candidates[0]) {
            dispatch(
              setNotification({
                message: `Proxied Jupyter URL not reachable. Using ${candidate}`,
                type: "warning",
              }),
            );
          }
          console.log("Using Jupyter URL:", candidate);
          setJupyterUrl(candidate);
          setEditableUrl(candidate);
          break;
        }
      } catch (error) {
        console.error("Error fetching Jupyter URL:", error);
        dispatch(
          setNotification({
            message: "Failed to fetch the Jupyter URL from the server.",
            type: "error",
          }),
        );
      }
    };
    fetchNotebookUrl();
  }, [hostIP, hostPort]);

  const handleUrlChange = (event) => {
    setEditableUrl(event.target.value);
  };

  const handleUrlSubmit = (event) => {
    if (event.key === "Enter" || event.type === "click") {
      setJupyterUrl(editableUrl);
      setIsEditing(false);
      setIframeKey((prev) => prev + 1); // Force iframe reload
    }
  };

  const handleRefresh = () => {
    // Reload the iframe by updating the key
    setIframeKey((prev) => prev + 1);
  };

  const handleOpenInNewTab = () => {
    window.open(jupyterUrl, "_blank");
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Top-Bar with URL editor and controls */}
      <AppBar position="static" sx={{ flex: "0 0 auto" }}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 0, marginRight: 2 }}>
            Jupyter Executor
          </Typography>

          {/* Editable URL field */}
          <Box sx={{ flexGrow: 1, display: "flex", alignItems: "center" }}>
            {isEditing ? (
              <TextField
                fullWidth
                size="small"
                value={editableUrl}
                onChange={handleUrlChange}
                onKeyDown={handleUrlSubmit}
                onBlur={() => setIsEditing(false)}
                autoFocus
                placeholder="Enter Jupyter URL..."
                variant="outlined"
                sx={{
                  backgroundColor: "white",
                  borderRadius: 1,
                  "& .MuiOutlinedInput-root": {
                    color: "black",
                  },
                }}
              />
            ) : (
              <Typography
                variant="body2"
                sx={{
                  flexGrow: 1,
                  cursor: "pointer",
                  backgroundColor: "rgba(255,255,255,0.1)",
                  padding: "8px 12px",
                  borderRadius: 1,
                  fontFamily: "monospace",
                  fontSize: "0.85rem",
                }}
                onClick={() => setIsEditing(true)}
              >
                {editableUrl || "Loading..."}
              </Typography>
            )}
          </Box>

          {/* Action buttons */}
          <Tooltip title="Edit URL">
            <IconButton
              color="inherit"
              onClick={() => setIsEditing(true)}
              sx={{ marginLeft: 1 }}
            >
              <Edit />
            </IconButton>
          </Tooltip>

          <Tooltip title="Refresh iframe">
            <IconButton color="inherit" onClick={handleRefresh}>
              <Refresh />
            </IconButton>
          </Tooltip>

          {jupyterUrl && (
            <Tooltip title="Open in new tab">
              <IconButton color="inherit" onClick={handleOpenInNewTab}>
                <OpenInNew />
              </IconButton>
            </Tooltip>
          )}
        </Toolbar>
      </AppBar>

      {/* Notebook (iframe) */}
      <Box sx={{ width: "100%", flex: "1 1 auto", minHeight: 0 }}>
        {jupyterUrl ? (
          <iframe
            key={iframeKey}
            src={jupyterUrl}
            style={{
              width: "100%",
              height: "100%",
              display: "block",
              boxSizing: "border-box",
              backgroundColor: "white",
              border: "none",
            }}
            title="Jupyter Notebook"
            onLoad={() => console.log("iframe loaded successfully")}
            onError={(e) => console.error("iframe load error:", e)}
          />
        ) : (
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              height: "100%",
              fontSize: "1.2rem",
              color: "gray",
            }}
          >
            Loading Jupyter Notebook...
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default JupyterExecutor;
