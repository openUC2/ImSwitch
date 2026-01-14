// src/components/AppManager/AppManager.jsx
// Main App Manager component - provides overview and control of all ImSwitch applications
// Inspired by BlueOS but designed for microscopy workflow management

import React, { useMemo } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  TextField,
  InputAdornment,
  Chip,
  Grid,
  Container,
  AppBar,
  Toolbar,
  IconButton,
  Tabs,
  Tab,
  Paper,
  Tooltip,
} from "@mui/material";
import {
  Search as SearchIcon,
  Apps as AppsIcon,
  Star as StarIcon,
  Code as CodeIcon,
  Computer as ComputerIcon,
  GridView as GridViewIcon,
  Refresh as RefreshIcon,
  PlayArrow as PlayArrowIcon,
} from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";

// Redux imports
import {
  selectEnabledApps,
  selectSearchQuery,
  selectSelectedCategory,
  selectAppStats,
  toggleApp,
  setSearchQuery,
  setSelectedCategory,
  resetToDefaults,
  enableCategory,
  disableCategory,
} from "../../state/slices/appManagerSlice";

// App registry imports
import {
  APP_REGISTRY,
  APP_CATEGORIES,
  getAppsByCategory,
  searchApps,
} from "../../constants/appRegistry";

// Category metadata for UI
const CATEGORY_INFO = {
  all: {
    label: "All Apps",
    icon: AppsIcon,
    color: "#2196f3",
    description: "All available applications",
  },
  [APP_CATEGORIES.ESSENTIALS]: {
    label: "Essentials",
    icon: StarIcon,
    color: "#ff9800",
    description: "Core microscopy components (always enabled)",
  },
  [APP_CATEGORIES.APPS]: {
    label: "Applications",
    icon: AppsIcon,
    color: "#4caf50",
    description: "Microscopy applications and tools",
  },
  [APP_CATEGORIES.CODING]: {
    label: "Coding",
    icon: CodeIcon,
    color: "#f44336",
    description: "Development and debugging tools",
  },
  [APP_CATEGORIES.SYSTEM]: {
    label: "System",
    icon: ComputerIcon,
    color: "#ff5722",
    description: "System configuration and utilities",
  },
};

/**
 * Individual app card component
 */
const AppCard = ({ app, isEnabled, onToggle, onLaunch }) => {
  const theme = useTheme();
  const IconComponent = app.icon;

  return (
    <Card
      elevation={2}
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        transition: "all 0.2s ease-in-out",
        "&:hover": {
          elevation: 4,
          transform: "translateY(-2px)",
        },
        opacity: isEnabled ? 1 : 0.7,
        borderLeft: `4px solid ${
          CATEGORY_INFO[app.category]?.color || theme.palette.primary.main
        }`,
      }}
    >
      <CardContent
        sx={{
          flexGrow: 1,
          cursor: "pointer",
          "&:hover": {
            backgroundColor: (theme) =>
              theme.palette.mode === "dark"
                ? "rgba(255, 255, 255, 0.05)"
                : "rgba(0, 0, 0, 0.02)",
          },
        }}
        onClick={() => onLaunch && onLaunch(app)}
      >
        <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
          <IconComponent
            sx={{
              fontSize: 32,
              mr: 2,
              color: isEnabled
                ? CATEGORY_INFO[app.category]?.color
                : theme.palette.text.secondary,
            }}
          />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" component="h3" gutterBottom>
              {app.name}
            </Typography>
            <Chip
              label={CATEGORY_INFO[app.category]?.label || app.category}
              size="small"
              sx={{
                backgroundColor:
                  CATEGORY_INFO[app.category]?.color ||
                  theme.palette.primary.main,
                color: "white",
                fontSize: "0.75rem",
              }}
            />
          </Box>
        </Box>

        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
            lineHeight: 1.4,
            minHeight: "4.2em", // 3 lines * 1.4 line height
          }}
        >
          {app.description}
        </Typography>

        {app.keywords.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography
              variant="caption"
              color="text.secondary"
              gutterBottom
              display="block"
            >
              Keywords:
            </Typography>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
              {app.keywords.slice(0, 4).map((keyword, index) => (
                <Chip
                  key={index}
                  label={keyword}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: "0.7rem", height: 20 }}
                />
              ))}
              {app.keywords.length > 4 && (
                <Chip
                  label={`+${app.keywords.length - 4}`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: "0.7rem", height: 20 }}
                />
              )}
            </Box>
          </Box>
        )}
      </CardContent>

      <CardActions sx={{ p: 2, pt: 0 }}>
        <FormControlLabel
          control={
            <Switch
              checked={isEnabled}
              onChange={() => !app.essential && onToggle(app.id)}
              disabled={app.essential}
              color="primary"
            />
          }
          label={
            app.essential
              ? "Essential (Always Available)"
              : isEnabled
              ? "Show in Navigation"
              : "Hidden from Navigation"
          }
          sx={{
            flexGrow: 1,
            "& .MuiFormControlLabel-label": {
              fontSize: "0.875rem",
              color: app.essential ? theme.palette.text.secondary : "inherit",
            },
          }}
        />

        <Tooltip title={`Launch ${app.name}`}>
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation(); // Prevent triggering card click
              onLaunch && onLaunch(app);
            }}
            sx={{
              color:
                CATEGORY_INFO[app.category]?.color ||
                theme.palette.primary.main,
              "&:hover": {
                backgroundColor: `${
                  CATEGORY_INFO[app.category]?.color ||
                  theme.palette.primary.main
                }20`,
              },
            }}
          >
            <PlayArrowIcon />
          </IconButton>
        </Tooltip>
      </CardActions>
    </Card>
  );
};

/**
 * Main App Manager component
 */
const AppManager = ({ onNavigateToApp }) => {
  const dispatch = useDispatch();

  // Redux state
  const enabledApps = useSelector(selectEnabledApps);
  const searchQuery = useSelector(selectSearchQuery);
  const selectedCategory = useSelector(selectSelectedCategory);
  const stats = useSelector(selectAppStats);

  // Computed data
  const filteredApps = useMemo(() => {
    let apps = Object.values(APP_REGISTRY);

    // Apply search filter
    if (searchQuery.trim()) {
      apps = searchApps(searchQuery);
    }

    // Apply category filter
    if (selectedCategory !== "all") {
      apps = apps.filter((app) => app.category === selectedCategory);
    }

    return apps;
  }, [searchQuery, selectedCategory]);

  // Event handlers
  const handleToggleApp = (appId) => {
    dispatch(toggleApp(appId));
  };

  const handleLaunchApp = (app) => {
    // Navigate to the app if navigation handler is provided
    if (onNavigateToApp && app.pluginId) {
      onNavigateToApp(app.pluginId);
    }
  };

  const handleSearchChange = (event) => {
    dispatch(setSearchQuery(event.target.value));
  };

  const handleCategoryChange = (event, newValue) => {
    dispatch(setSelectedCategory(newValue));
  };

  const handleResetToDefaults = () => {
    dispatch(resetToDefaults());
  };

  const handleEnableAllInCategory = () => {
    if (selectedCategory !== "all") {
      dispatch(enableCategory(selectedCategory));
    }
  };

  const handleDisableAllInCategory = () => {
    if (
      selectedCategory !== "all" &&
      selectedCategory !== APP_CATEGORIES.ESSENTIALS
    ) {
      dispatch(disableCategory(selectedCategory));
    }
  };

  // Category statistics
  const categoryStats = useMemo(() => {
    const stats = {};
    Object.values(APP_CATEGORIES).forEach((category) => {
      const appsInCategory = getAppsByCategory(category);
      const enabledInCategory = appsInCategory.filter((app) =>
        enabledApps.includes(app.id)
      );
      stats[category] = {
        total: appsInCategory.length,
        enabled: enabledInCategory.length,
      };
    });
    return stats;
  }, [enabledApps]);

  return (
    <Box sx={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header with explanation */}
      <AppBar position="static" elevation={1}>
        <Toolbar>
          <GridViewIcon sx={{ mr: 2 }} />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="h6" component="h1">
              Customize Your Workspace
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              Choose which tools appear in your navigation drawer
            </Typography>
          </Box>

          <Typography variant="body2" sx={{ mr: 2 }}>
            {stats.enabledCount} of {stats.totalApps + 2} apps enabled
            {/* 2 additional for essential apps */}
          </Typography>

          <Tooltip title="Reset to defaults">
            <IconButton color="inherit" onClick={handleResetToDefaults}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      {/* Quick Help Banner */}
      <Paper
        elevation={0}
        sx={{
          bgcolor: (theme) =>
            theme.palette.mode === "dark"
              ? "rgba(156, 39, 176, 0.1)"
              : "rgba(156, 39, 176, 0.05)",
          borderBottom: 1,
          borderColor: "divider",
        }}
      >
        <Container maxWidth="lg" sx={{ py: 1 }}>
          <Typography
            variant="body2"
            sx={{ textAlign: "center", color: "text.secondary" }}
          >
            💡 <strong>Tip:</strong> Toggle apps on/off to customize your
            navigation drawer. Enabled apps will appear in your sidebar for
            quick access.
          </Typography>
        </Container>
      </Paper>

      {/* Search and Controls */}
      <Paper elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Container maxWidth="lg" sx={{ py: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                placeholder="Search apps by name, description, or keywords..."
                value={searchQuery}
                onChange={handleSearchChange}
                variant="outlined"
                size="small"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                {selectedCategory !== "all" &&
                  selectedCategory !== APP_CATEGORIES.ESSENTIALS && (
                    <>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={handleEnableAllInCategory}
                      >
                        Enable All
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        color="error"
                        onClick={handleDisableAllInCategory}
                      >
                        Disable All
                      </Button>
                    </>
                  )}
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Paper>

      {/* Category Tabs */}
      <Paper elevation={0} sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Container maxWidth="lg">
          <Tabs
            value={selectedCategory}
            onChange={handleCategoryChange}
            variant="scrollable"
            scrollButtons="auto"
          >
            {Object.entries(CATEGORY_INFO).map(([key, info]) => {
              const IconComponent = info.icon;
              const count =
                key === "all"
                  ? stats.totalApps
                  : categoryStats[key]?.total || 0;
              const enabled =
                key === "all"
                  ? stats.enabledCount
                  : categoryStats[key]?.enabled || 0;

              return (
                <Tab
                  key={key}
                  value={key}
                  sx={{
                    minWidth: 120, // Ensure minimum width for proper spacing
                    px: 2, // Add horizontal padding
                  }}
                  label={
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5, // Increase gap between elements
                        minWidth: 0, // Allow text to shrink if needed
                      }}
                    >
                      <IconComponent fontSize="small" />
                      <Typography
                        variant="body2"
                        sx={{
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {info.label}
                      </Typography>
                      <Chip
                        label={`${enabled}/${count}`}
                        size="small"
                        color="primary"
                        variant="outlined"
                        sx={{
                          height: 20,
                          fontSize: "0.75rem",
                          minWidth: 40,
                        }}
                      />
                    </Box>
                  }
                />
              );
            })}
          </Tabs>
        </Container>
      </Paper>

      {/* App Grid */}
      <Box sx={{ flexGrow: 1, bgcolor: "background.default" }}>
        <Container maxWidth="lg" sx={{ py: 3 }}>
          {filteredApps.length === 0 ? (
            <Paper sx={{ p: 4, textAlign: "center" }}>
              <SearchIcon
                sx={{ fontSize: 48, color: "text.secondary", mb: 2 }}
              />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No apps found
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {searchQuery
                  ? "Try adjusting your search terms"
                  : "No apps available in this category"}
              </Typography>
            </Paper>
          ) : (
            <Grid container spacing={3}>
              {filteredApps.map((app) => (
                <Grid item xs={12} sm={6} lg={4} key={app.id}>
                  <AppCard
                    app={app}
                    isEnabled={enabledApps.includes(app.id)}
                    onToggle={handleToggleApp}
                    onLaunch={handleLaunchApp}
                  />
                </Grid>
              ))}
            </Grid>
          )}
        </Container>
      </Box>
    </Box>
  );
};

export default AppManager;
