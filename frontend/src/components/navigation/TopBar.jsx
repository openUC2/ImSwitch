import { Toolbar, Typography, IconButton, Box } from "@mui/material";
import MenuOpen from "@mui/icons-material/MenuOpen";
import { useSelector } from "react-redux";
import SettingsMenu from "./SettingsMenu.jsx";
import StorageButton from "../StorageButton.js";
import * as uc2Slice from "../../state/slices/UC2Slice.js";

/**
 * ImSwitch TopBar Component
 */
const TopBar = ({
  isMobile,
  sidebarVisible,
  setSidebarVisible,
  selectedPlugin,
  onSettingsNavigate,
  onFileManagerRefresh,
  onStorageChange,
}) => {
  // Get backend connection status
  const uc2State = useSelector(uc2Slice.getUc2State);
  const isBackendConnected = uc2State.backendConnected;
  return (
    <Box
      sx={(theme) => ({
        width: "100%",
        background: theme.palette.background.paper,
        color: theme.palette.text.primary,
        borderBottom: "1px solid",
      })}
    >
      <Toolbar
        sx={{ width: "100%", display: "flex", alignItems: "center", px: 2 }}
      >
        <>
          {isMobile ? (
            <IconButton
              edge="start"
              color="inherit"
              onClick={() => setSidebarVisible(!sidebarVisible)}
              sx={{ mr: 2 }}
            >
              <MenuOpen sx={{ transform: "scaleX(-1)" }} />
            </IconButton>
          ) : (
            <></>
          )}
        </>

        <Typography
          variant="h6"
          sx={{
            flexGrow: 1,
            fontWeight: "bold",
            fontSize: isMobile ? "1rem" : "1.25rem",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {isMobile ? selectedPlugin : `ImSwitch UI - ${selectedPlugin}`}
        </Typography>

        <StorageButton
          onFileManagerRefresh={onFileManagerRefresh}
          onStorageChange={onStorageChange}
          disabled={!isBackendConnected}
        />
        <SettingsMenu onNavigate={onSettingsNavigate} />
      </Toolbar>
    </Box>
  );
};

export default TopBar;
