import { useDispatch, useSelector } from "react-redux";
import { Switch, Typography } from "@mui/material";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import { toggleTheme, getThemeState } from "../state/slices/ThemeSlice";

function ThemeSwitcher({ isMobile }) {
  const dispatch = useDispatch();
  const { isDarkMode } = useSelector(getThemeState);

  const handleToggle = () => {
    dispatch(toggleTheme());
  };

  return (
    <div style={{ display: "flex", alignItems: "center", minHeight: 40 }}>
      <>
        {isDarkMode ? (
          <DarkModeIcon sx={{ mr: 1 }} />
        ) : (
          <LightModeIcon sx={{ mr: 1 }} />
        )}
        <Typography variant="body2" sx={{ fontWeight: "bold", marginRight: 1 }}>
          {isDarkMode ? "Dark" : "Light"}
        </Typography>
      </>
      <Switch
        checked={isDarkMode}
        onChange={handleToggle}
        color="default"
        inputProps={{ "aria-label": "toggle theme" }}
        sx={{
          mx: isMobile ? 0 : 1,
          "& .MuiSwitch-thumb": {
            width: 20,
            height: 20,
          },
          "& .MuiSwitch-track": {
            minWidth: 34,
            height: 14,
          },
        }}
      />
    </div>
  );
}

export default ThemeSwitcher;
