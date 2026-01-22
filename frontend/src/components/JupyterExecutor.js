import React, { useEffect, useState } from "react";
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  TextField,
  Box,
  IconButton,
  Tooltip,
  Chip
} from "@mui/material";
import { OpenInNew, Refresh, Edit } from "@mui/icons-material";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

const JupyterExecutor = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  const [jupyterUrl, setJupyterUrl] = useState(null);
  const [editableUrl, setEditableUrl] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [urlStatus, setUrlStatus] = useState({ proxied: null, direct: null });
  const [iframeKey, setIframeKey] = useState(0);

  // Test if a URL is accessible
  const testUrl = async (url) => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const response = await fetch(url, {
        method: 'HEAD',
        signal: controller.signal,
        mode: 'no-cors' // Allow checking if server responds
      });
      
      clearTimeout(timeoutId);
      return true;
    } catch (error) {
      return false;
    }
  };

  useEffect(() => {
    const fetchNotebookUrl = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/jupyternotebookurl`
        );
        const data = await response.json();
        const notebookUrl = data["url"]; // e.g., http://192.168.1.100:8888/jupyter

        // Extract the path from the notebook URL
        const urlObj = new URL(notebookUrl);
        const jupyterPath = urlObj.pathname; // e.g., /jupyter
        const jupyterPort = urlObj.port; // e.g., 8888

        // Construct both possible URLs:
        // 1. Proxied URL through the ImSwitch API server (Caddy reverse proxy in Docker)
        const proxiedUrl = `${hostIP}:${hostPort}${jupyterPath}`;
        
        // 2. Direct URL to Jupyter server (for local development)
        const directUrl = notebookUrl.replace(
          /https?:\/\/[^:\/]+/,
          hostIP.replace(/https?:\/\//, 'http://')
        );

        // Test both URLs to see which one works
        console.log("Testing Jupyter URLs...");
        console.log("Proxied URL:", proxiedUrl);
        console.log("Direct URL:", directUrl);

        const [proxiedWorks, directWorks] = await Promise.all([
          testUrl(proxiedUrl),
          testUrl(directUrl)
        ]);

        setUrlStatus({ proxied: proxiedWorks, direct: directWorks });

        // Prefer proxied URL if it works, otherwise use direct URL
        let finalUrl;
        if (proxiedWorks) {
          finalUrl = proxiedUrl;
          console.log("✓ Using proxied Jupyter URL:", proxiedUrl);
        } else if (directWorks) {
          finalUrl = directUrl;
          console.log("✓ Using direct Jupyter URL:", directUrl);
        } else {
          // If neither works, default to direct URL and let user adjust
          finalUrl = directUrl;
          console.warn("⚠ Neither URL responded, defaulting to:", directUrl);
        }

        setJupyterUrl(finalUrl);
        setEditableUrl(finalUrl);
      } catch (error) {
        console.error("Error fetching Jupyter URL:", error);
      }
    };
    fetchNotebookUrl();
  }, [hostIP, hostPort]);

  const handleUrlChange = (event) => {
    setEditableUrl(event.target.value);
  };

  const handleUrlSubmit = (event) => {
    if (event.key === 'Enter' || event.type === 'click') {
      setJupyterUrl(editableUrl);
      setIsEditing(false);
      setIframeKey(prev => prev + 1); // Force iframe reload
    }
  };

  const handleRefresh = () => {
    // Reload the iframe by updating the key
    setIframeKey(prev => prev + 1);
  };

  const handleOpenInNewTab = () => {
    window.open(jupyterUrl, '_blank');
  };

  return (
    <>
      {/* Top-Bar with URL editor and controls */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 0, marginRight: 2 }}>
            Jupyter Executor
          </Typography>
          
          {/* URL Status Indicators */}
          {urlStatus.proxied !== null && (
            <Tooltip title="Proxied URL (through ImSwitch)">
              <Chip 
                label="Proxied" 
                size="small"
                color={urlStatus.proxied ? "success" : "default"}
                sx={{ marginRight: 1 }}
              />
            </Tooltip>
          )}
          {urlStatus.direct !== null && (
            <Tooltip title="Direct URL (to Jupyter port)">
              <Chip 
                label="Direct" 
                size="small"
                color={urlStatus.direct ? "success" : "default"}
                sx={{ marginRight: 2 }}
              />
            </Tooltip>
          )}

          {/* Editable URL field */}
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
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
                  backgroundColor: 'white', 
                  borderRadius: 1,
                  '& .MuiOutlinedInput-root': {
                    color: 'black'
                  }
                }}
              />
            ) : (
              <Typography 
                variant="body2" 
                sx={{ 
                  flexGrow: 1, 
                  cursor: 'pointer',
                  backgroundColor: 'rgba(255,255,255,0.1)',
                  padding: '8px 12px',
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.85rem'
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
            <IconButton 
              color="inherit" 
              onClick={handleRefresh}
            >
              <Refresh />
            </IconButton>
          </Tooltip>

          {jupyterUrl && (
            <Tooltip title="Open in new tab">
              <IconButton 
                color="inherit" 
                onClick={handleOpenInNewTab}
              >
                <OpenInNew />
              </IconButton>
            </Tooltip>
          )}
        </Toolbar>
      </AppBar>

      {/* Notebook (iframe) */}
      <div style={{ width: "100%", height: "calc(100vh - 64px)", position: "relative" }}>
        {jupyterUrl ? (
          <iframe
            key={iframeKey}
            src={jupyterUrl}
            style={{ width: "100%", height: "100%", border: "none" }}
            title="Jupyter Notebook"
          />
        ) : (
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              height: '100%',
              fontSize: '1.2rem',
              color: 'gray'
            }}
          >
            Loading Jupyter Notebook...
          </Box>
        )}
      </div>
    </>
  );
};

export default JupyterExecutor;
