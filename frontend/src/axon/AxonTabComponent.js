import React, { useState, useEffect } from "react";
import { useDispatch } from "react-redux";
import LiveViewControlWrapper from "./LiveViewControlWrapper";
import GenericTabBar from "./GenericTabBar";
import WellSelectorComponent from "./WellSelectorComponent";
import PointListEditorComponent from "./PointListEditorComponent";
import WebSocketComponent from "./WebSocketComponent";
import PositionViewComponent from "./PositionViewComponent";
import ParameterEditorWrapper from "./ParameterEditorWrapper";
import FocusLockMiniController from "../components/FocusLockMiniController";
import Frame3DViewerPanel from "./Frame3DViewerPanel.jsx";
import PictureInPicture, { PiPToggleButton } from "./PictureInPicture";
import OverviewScanTab from "./OverviewScanTab";
import OverviewRegistrationWizard from "./OverviewRegistrationWizard.js";
import DetectorToggle from "./DetectorToggle";
import WellPlateWorkspace from "./wellplate2/WellPlateWorkspace";
import createAxiosInstance from "../backendapi/createAxiosInstance";
import { setNotification } from "../state/slices/NotificationSlice";
import * as detectorParametersSlice from "../state/slices/DetectorParametersSlice";
import fetchObjectiveControllerGetStatus from "../middleware/fetchObjectiveControllerGetStatus.js";

// Feature flag for the renovated "Linked Workspace" WellPlate layout
// (single viewport tab strip on the left + the existing experiment inspector
// always present on the right). Flip to false to fall back to the legacy
// two-tab-bar layout kept below for rollback.
const USE_NEW_WELLPLATE_LAYOUT = true;

const AxonTabComponent = () => {
  const dispatch = useDispatch();
  // PiP (picture-in-picture) floating live preview
  const [pipVisible, setPipVisible] = useState(false);

  // Opening the WellPlate view must always refresh the objective state so the
  // current pixel size / FOV is applied to the tiling, overlap and freehand
  // step-size maths. This lives on the container (not the Well Selector tab)
  // because GenericTabBar only mounts the active tab — if the user last left
  // the view on "Points" or "Parameter", the Well Selector's own mount fetch
  // would never run and the objective would go un-queried.
  useEffect(() => {
    fetchObjectiveControllerGetStatus(dispatch);
  }, [dispatch]);

  // Ensure WellPlate starts with detector auto-exposure disabled.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const api = createAxiosInstance();
        await api.get(`/SettingsController/setDetectorMode?isAuto=false`);
        if (cancelled) return;

        // Keep frontend mode state in sync until the next WS update arrives.
        dispatch(
          detectorParametersSlice.updateParameter({
            key: "mode",
            value: "manual",
          }),
        );

        dispatch(
          setNotification({
            message:
              "WellPlate started: Live view auto-exposure has been disabled.",
            type: "info",
            autoHideDuration: 3500,
          }),
        );
      } catch (error) {
        if (cancelled) return;
        dispatch(
          setNotification({
            message:
              "WellPlate started, but live view auto-exposure could not be disabled.",
            type: "warning",
            autoHideDuration: 4500,
          }),
        );
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [dispatch]);

  return (
    <div style={{ width: "100%" }}>
      {/* PiP toggle button – always visible in the top-right corner */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          padding: "2px 8px 0 0",
        }}
      >
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

      {USE_NEW_WELLPLATE_LAYOUT ? (
        /* Renovated layout: viewport tab strip + always-on experiment inspector */
        <WellPlateWorkspace />
      ) : (
        /* Legacy layout: two competing tab bars (kept for rollback) */
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
                "Overview Scan",
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
      )}

      {/* Overview Registration Wizard — mounted here so it's always available
          regardless of which viewport is active */}
      <OverviewRegistrationWizard />
    </div>
  );
};

export default AxonTabComponent;
