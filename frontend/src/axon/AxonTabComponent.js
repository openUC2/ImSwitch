import React, { useState } from "react";
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
import ObjectiveController from "../components/ObjectiveController";
import ResizablePanel from "./ResizablePanel"; //<ResizablePanel></ResizablePanel> performace issues :/
import ObjectiveSwitcher from "../components/ObjectiveSwitcher";
import FocusLockMiniController from "../components/FocusLockMiniController";
import PictureInPicture, { PiPToggleButton } from "./PictureInPicture";

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
            ]}
          >
            <WellSelectorComponent />
            <LiveViewControlWrapper />
            <ParameterEditorWrapper />

            <PointListEditorComponent />
            <div style={{ display: "flex" }}>
              <WebSocketComponent />
              <PositionViewComponent />
            </div>
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
              "Objective", 
              "Focus Lock"
            ]}
          >
            <LiveViewControlWrapper />
            {/* <ZarrTileViewController /> */}
            <PointListEditorComponent />
            <ParameterEditorWrapper />
            {/*<ExperimentComponent />*/}
            <ObjectiveSwitcher />
            <FocusLockMiniController />
          </GenericTabBar>
        </div>
      </div>
    </div>
  );
};

export default AxonTabComponent;
