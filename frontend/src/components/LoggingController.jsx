import { useState, useEffect } from "react";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";
import * as uc2Slice from "../state/slices/UC2Slice.js";
import {
  Box,
  Typography,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Alert,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  Divider,
} from "@mui/material";
import {
  InsertDriveFile,
  Download,
  Refresh,
  ErrorOutline,
  CheckCircle,
  Description,
} from "@mui/icons-material";

/**
 * LoggingController component
 * Interface for viewing and downloading ImSwitch log files
 */
export default function LoggingController() {
  // Get connection settings from Redux
  const { ip: hostIP, apiPort: hostPort } = useSelector(
    getConnectionSettingsState
  );

  // Get backend connection status
  const uc2State = useSelector(uc2Slice.getUc2State);
  const isBackendConnected = uc2State.backendConnected;

  // Component state
  const [logFiles, setLogFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);

  const baseUrl = `${hostIP}:${hostPort}/imswitch/api`;

  /**
   * Fetch the list of log files from the backend
   */
  const fetchLogFiles = async () => {
    if (!isBackendConnected) {
      setError("Backend not connected");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${baseUrl}/LogController/listLogFiles`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Log files received:", data);

      // Sort by filename (newest first - reverse alphanumerical)
      const sortedFiles = (data.log_files || []).sort((a, b) =>
        b.filename.localeCompare(a.filename)
      );

      setLogFiles(sortedFiles);
    } catch (err) {
      console.error("Error fetching log files:", err);
      setError(`Failed to fetch log files: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Download a specific log file
   */
  const downloadLogFile = async (filename) => {
    if (!isBackendConnected) {
      setError("Backend not connected");
      return;
    }

    try {
      setSelectedFile(filename);
      const response = await fetch(
        `${baseUrl}/LogController/downloadLogFile?filename=${encodeURIComponent(
          filename
        )}`,
        {
          method: "GET",
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get the blob data
      const blob = await response.blob();

      // Create a temporary URL and trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      console.log(`Downloaded file: ${filename}`);
    } catch (err) {
      console.error("Error downloading log file:", err);
      setError(`Failed to download file: ${err.message}`);
    } finally {
      setSelectedFile(null);
    }
  };

  /**
   * Format file size for display
   */
  const formatFileSize = (sizeStr) => {
    const size = parseInt(sizeStr, 10);
    if (isNaN(size)) return "Unknown size";

    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  };

  /**
   * Format date for display
   */
  const formatDate = (isoDate) => {
    try {
      const date = new Date(isoDate);
      return date.toLocaleString();
    } catch {
      return isoDate;
    }
  };

  // Fetch log files on component mount and when backend connection changes
  useEffect(() => {
    if (isBackendConnected) {
      fetchLogFiles();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isBackendConnected]);

  return (
    <Box sx={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Box>
          <Typography variant="h4" gutterBottom>
            ImSwitch Logging
          </Typography>
          <Typography variant="body2" color="text.secondary">
            View and download ImSwitch log files
          </Typography>
        </Box>
        <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
          <Chip
            icon={isBackendConnected ? <CheckCircle /> : <ErrorOutline />}
            label={isBackendConnected ? "Connected" : "Disconnected"}
            color={isBackendConnected ? "success" : "error"}
            variant="outlined"
            size="small"
          />
          <Tooltip title="Refresh log file list">
            <IconButton
              onClick={fetchLogFiles}
              disabled={!isBackendConnected || loading}
              color="primary"
            >
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Error display */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Backend not connected warning */}
      {!isBackendConnected && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Backend is not connected. Please configure the connection in settings.
        </Alert>
      )}

      {/* Loading state */}
      {loading && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            py: 4,
          }}
        >
          <CircularProgress />
        </Box>
      )}

      {/* Log files list */}
      {!loading && isBackendConnected && (
        <Card sx={{ flexGrow: 1, overflow: "auto" }}>
          <CardContent sx={{ p: 0, "&:last-child": { pb: 0 } }}>
            {logFiles.length === 0 ? (
              <Box sx={{ p: 4, textAlign: "center" }}>
                <Description sx={{ fontSize: 60, color: "text.secondary", mb: 2 }} />
                <Typography variant="body1" color="text.secondary">
                  No log files found
                </Typography>
              </Box>
            ) : (
              <List sx={{ width: "100%", bgcolor: "background.paper" }}>
                {logFiles.map((file, index) => (
                  <Box key={file.filename}>
                    {index > 0 && <Divider />}
                    <ListItem
                      disablePadding
                      secondaryAction={
                        <Tooltip title="Download log file">
                          <IconButton
                            edge="end"
                            onClick={() => downloadLogFile(file.filename)}
                            disabled={selectedFile === file.filename}
                          >
                            {selectedFile === file.filename ? (
                              <CircularProgress size={24} />
                            ) : (
                              <Download />
                            )}
                          </IconButton>
                        </Tooltip>
                      }
                    >
                      <ListItemButton
                        onClick={() => downloadLogFile(file.filename)}
                        disabled={selectedFile === file.filename}
                      >
                        <ListItemIcon>
                          <InsertDriveFile color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary={file.filename}
                          secondary={
                            <span>
                              {formatFileSize(file.size)} • Modified:{" "}
                              {formatDate(file.modified)}
                            </span>
                          }
                        />
                      </ListItemButton>
                    </ListItem>
                  </Box>
                ))}
              </List>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
