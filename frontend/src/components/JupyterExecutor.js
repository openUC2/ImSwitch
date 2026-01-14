import React, { useEffect, useState } from "react";
import { AppBar, Toolbar, Typography, Button } from "@mui/material";
import { useSelector } from "react-redux";
import { getConnectionSettingsState } from "../state/slices/ConnectionSettingsSlice";

const JupyterExecutor = () => {
  // Get connection settings from Redux
  const connectionSettings = useSelector(getConnectionSettingsState);
  const hostIP = connectionSettings.ip;
  const hostPort = connectionSettings.apiPort;
  const [jupyterUrl, setJupyterUrl] = useState(null);

  useEffect(() => {
    const fetchNotebookUrl = async () => {
      try {
        const response = await fetch(
          `${hostIP}:${hostPort}/imswitch/api/jupyternotebookurl`
        );
        const data = await response.json();
        const notebookUrl = data["url"];
        if (
          notebookUrl.includes("localhost") ||
          notebookUrl.includes("0.0.0.0") ||
          notebookUrl.includes(".local")
        ) {
          setJupyterUrl(notebookUrl);
        } else {
          const updatedUrl = notebookUrl.replace(
            /https?:\/\/[^:]+/,
            `${hostIP.replace("https://", "http://")}`
          );
          setJupyterUrl(updatedUrl);
        }
      } catch (error) {
        console.error("Error fetching Jupyter URL:", error);
      }
    };
    fetchNotebookUrl();
  }, [hostIP, hostPort]);

  return (
    <>
      {/* Top-Bar mit Link zum Jupyter Notebook */}
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Jupyter Executor
          </Typography>
          {jupyterUrl && (
            <Button color="inherit" href={jupyterUrl} target="_blank">
              Open Jupyter Lab
            </Button>
          )}
        </Toolbar>
      </AppBar>

      {/* Notebook (iframe) */}
      <div style={{ width: "100%", height: "100%", position: "relative" }}>
        {jupyterUrl ? (
          <iframe
            src={jupyterUrl}
            style={{ width: "100%", height: "100vh", border: "none" }}
            title="Jupyter Notebook"
          />
        ) : (
          <p>Loading Jupyter Notebook...</p>
        )}
      </div>
    </>
  );
};

export default JupyterExecutor;
