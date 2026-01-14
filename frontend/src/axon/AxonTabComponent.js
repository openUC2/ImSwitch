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

const AxonTabComponent = () => {
  return (
    <div style={{ width: "100%" }}>
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
              "Tile View",
              "Points",
              "Parameter",
              "Objective", 
              "Focus Lock"
            ]}
          >
            <LiveViewControlWrapper />
            <ZarrTileViewController/>
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
