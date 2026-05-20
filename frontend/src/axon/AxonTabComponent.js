import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import {
  ToggleButton,
  ToggleButtonGroup,
  Typography,
  Box,
} from "@mui/material";
import {
  Videocam as VideocamIcon,
} from "@mui/icons-material";
import LiveViewControlWrapper from "./LiveViewControlWrapper";
import TileViewComponent from "./TileViewComponent";
import ZarrTileViewController from "./ZarrTileView";
import GenericTabBar from "./GenericTabBar";
import WellSelectorComponent from "./WellSelectorComponent";
import PointListEditorComponent from "./PointListEditorComponent";
import WebSocketComponent from "./WebSocketComponent";
import PositionViewComponent from "./PositionViewComponent";
import ParameterEditorWrapper from "./ParameterEditorWrapper";
import ExperimentComponent from "./ExperimentComponent";
import ResizablePanel from "./ResizablePanel"; //<ResizablePanel></ResizablePanel> performace issues :/
import FocusLockMiniController from "../components/FocusLockMiniController";
import Frame3DViewerPanel from "./Frame3DViewerPanel.jsx";
import PictureInPicture, { PiPToggleButton } from "./PictureInPicture";
import OverviewScanTab from "./OverviewScanTab";
import OverviewRegistrationWizard from "./OverviewRegistrationWizard.js";
import * as liveViewSlice from "../state/slices/LiveViewSlice.js";
import * as liveStreamSlice from "../state/slices/LiveStreamSlice.js";
import apiLiveViewControllerStartLiveView from "../backendapi/apiLiveViewControllerStartLiveView";
import apiLiveViewControllerStopLiveView from "../backendapi/apiLiveViewControllerStopLiveView";

/**
 * DetectorToggle - Camera selector for switching between detectors
 * (e.g. overview vs widefield) within the WellPlate app.
 */
const DetectorToggle = () => {
  const dispatch = useDispatch();
  const liveViewState = useSelector(liveViewSlice.getLiveViewState);
  const liveStreamState = useSelector(liveStreamSlice.getLiveStreamState);
  const detectors = liveViewState.detectors || [];
  const activeTab = liveViewState.activeTab || 0;

  if (detectors.length < 2) return null; // No toggle needed for single camera

  const handleSwitch = async (_event, newIdx) => {
    if (newIdx === null || newIdx === activeTab) return;
    dispatch(liveViewSlice.setActiveTab(newIdx));

    // Restart stream on new detector (mirrors LiveView.js behaviour)
    if (liveViewState.isStreamRunning) {
      try {
        await apiLiveViewControllerStopLiveView();
        await new Promise((r) => setTimeout(r, 200));
        const protocol = liveStreamState.imageFormat || "jpeg";
        const newDetectorName = detectors[newIdx] || null;
        const savedParams =
          newDetectorName && liveStreamState.perDetectorSettings?.[newDetectorName];
        const overrideParams =
          savedParams && savedParams.protocol === protocol ? savedParams : null;
        const result = await apiLiveViewControllerStartLiveView(
          newDetectorName,
          protocol,
          overrideParams,
        );
        if (result?.params && newDetectorName) {
          dispatch(
            liveStreamSlice.updateDetectorSettings({
              detectorName: newDetectorName,
              settings: result.params,
            }),
          );
        }
      } catch (err) {
        console.error("[DetectorToggle] Switch failed:", err);
      }
    }
  };

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
      <VideocamIcon fontSize="small" color="action" />
      <Typography variant="caption" color="text.secondary">Camera:</Typography>
      <ToggleButtonGroup
        value={activeTab}
        exclusive
        onChange={handleSwitch}
        size="small"
        sx={{ '& .MuiToggleButton-root': { textTransform: 'none', py: 0.25, px: 1.5 } }}
      >
        {detectors.map((name, idx) => (
          <ToggleButton key={name} value={idx}>
            {name}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
};

const AxonTabComponent = () => {
  // PiP (picture-in-picture) floating live preview
  const [pipVisible, setPipVisible] = useState(false);

  return (
    <div style={{ width: "100%" }}>
      {/* PiP toggle button – always visible in the top-right corner */}
      <div style={{ display: "flex", justifyContent: "flex-end", padding: "2px 8px 0 0" }}>
        <PiPToggleButton
          active={pipVisible}
          onClick={() => setPipVisible((v) => !v)}
        />
      </div>

      {/* Floating PiP overlay */}
      <PictureInPicture
        visible={pipVisible}
        onClose={() => setPipVisible(false)}
      />

      <div style={{ display: "flex" }}>
        <div style={{ flex: 3 }}>
          <GenericTabBar
            id="1"
            tabNames={[
              "Well Selector",
              "Live View",
              "Parameter",
              "Points",
              "State",
              "3D Twin",
            ]}
          >
            <WellSelectorComponent />
            <div>
              <DetectorToggle />
              <LiveViewControlWrapper />
            </div>
            <ParameterEditorWrapper />

            <PointListEditorComponent />
            <div style={{ display: "flex" }}>
              <WebSocketComponent />
              <PositionViewComponent />
            </div>
            <Frame3DViewerPanel />
          </GenericTabBar>
        </div>
        <div style={{ flex: 2 }}>
          <GenericTabBar
            id="2"
            tabNames={[
              "Live View",
//              "Tile View",
              "Points",
              "Parameter",
              "Focus Lock",
              "Overview Scan"
            ]}
          >
            <div>
              <DetectorToggle />
              <LiveViewControlWrapper />
            </div>
            {/* <ZarrTileViewController /> */}
            <PointListEditorComponent />
            <ParameterEditorWrapper />
            {/*<ExperimentComponent />*/}
            <FocusLockMiniController />
            <OverviewScanTab />
          </GenericTabBar>
        </div>
      </div>

      {/* Overview Registration Wizard — mounted here so it's always available
          regardless of which tab is active in the left GenericTabBar */}
      <OverviewRegistrationWizard />
    </div>
  );
};

export default AxonTabComponent;
