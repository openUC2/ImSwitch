// Kiosk WiFi page: embeds the device-admin "Internet Access" panel (the same
// external page WiFiController.jsx uses — ImSwitch itself has no WiFi API).
import { useState } from "react";
import { useSelector } from "react-redux";
import { Box, Button, Typography } from "@mui/material";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";

import MobilePage from "../components/MobilePage";
import { getConnectionSettingsState } from "../../state/slices/ConnectionSettingsSlice";

const MobileWifiPage = () => {
  const connectionSettings = useSelector(getConnectionSettingsState);
  const [reloadKey, setReloadKey] = useState(0);

  // The device-admin service runs on the plain host (port 80/443), not on the
  // ImSwitch API port — same URL scheme as components/WiFiController.jsx.
  const internetAccessUrl = `${connectionSettings.ip}/admin/panel/internet?nav=hidden&theme=dark`;

  return (
    <MobilePage
      title="WiFi & Network"
      subtitle="Connect the microscope to a wireless network"
      action={
        <Button
          variant="outlined"
          startIcon={<RefreshRoundedIcon />}
          onClick={() => setReloadKey((k) => k + 1)}
        >
          Reload
        </Button>
      }
      disablePadding
    >
      <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
        <iframe
          key={reloadKey}
          src={internetAccessUrl}
          title="Network configuration"
          style={{ flex: 1, width: "100%", border: "none", minHeight: 0 }}
        />
        <Typography
          variant="caption"
          color="text.disabled"
          sx={{ px: 2, py: 0.75, flexShrink: 0 }}
        >
          If this panel stays empty, the device-admin service is not reachable
          at {internetAccessUrl}
        </Typography>
      </Box>
    </MobilePage>
  );
};

export default MobileWifiPage;
