// Warns when a camera is not really there: it did not open at startup (a
// simulated camera shows test images, and recordings/jobs refuse to start) or
// the live stream stopped getting new frames (unplugged?). Restarting ImSwitch
// re-opens every camera, so a camera plugged in afterwards works again.
import React, { useEffect, useState } from "react";
import { Alert, Button, Stack } from "@mui/material";
import apiLiveViewControllerGetCameraHealth from "../backendapi/apiLiveViewControllerGetCameraHealth";
import apiUC2ConfigControllerRestartImSwitch from "../backendapi/apiUC2ConfigControllerRestartImSwitch";

const POLL_MS = 5000;

export default function CameraHealthBanner({ sx }) {
  const [problems, setProblems] = useState([]); // [[name, {message, placeholder, stalled}]]
  const [restarting, setRestarting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      apiLiveViewControllerGetCameraHealth()
        .then((all) => {
          if (!cancelled)
            setProblems(Object.entries(all || {}).filter(([, h]) => h.placeholder || h.stalled));
        })
        .catch(() => {}); // older backend or server restarting: show nothing
    load();
    const timer = setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const restart = async () => {
    if (!window.confirm("Restart ImSwitch? Running scans and recordings stop; the page reconnects after about a minute.")) return;
    setRestarting(true);
    await apiUC2ConfigControllerRestartImSwitch().catch(() => {}); // the connection drops by design
  };

  if (!problems.length) return null;
  return (
    <Stack spacing={1} sx={sx}>
      {problems.map(([name, h]) => (
        <Alert
          key={name}
          severity="warning"
          action={
            <Button color="inherit" size="small" disabled={restarting} onClick={restart}>
              {restarting ? "Restarting…" : "Restart ImSwitch"}
            </Button>
          }
        >
          {h.message}
          {h.placeholder && " Recordings, saved snaps and jobs that take images are refused until then."}
        </Alert>
      ))}
    </Stack>
  );
}
