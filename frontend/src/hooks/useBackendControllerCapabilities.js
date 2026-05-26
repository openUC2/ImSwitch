import { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  fetchAvailableControllers,
  selectAvailableControllers,
} from "../state/slices/BackendCapabilitiesSlice";

const useBackendControllerCapabilities = ({
  hostIP,
  apiPort,
  selectedPlugin,
  setSelectedPlugin,
}) => {
  const dispatch = useDispatch();
  const availableControllers = useSelector(selectAvailableControllers);

  useEffect(() => {
    dispatch(fetchAvailableControllers());
  }, [hostIP, apiPort, dispatch]);

  useEffect(() => {
    const hasObjectiveController = availableControllers.includes(
      "ObjectiveController",
    );
    const hasLEDMatrixController = availableControllers.includes(
      "LEDMatrixController",
    );
    const hasDPCController = availableControllers.includes("DPCController");
    const hasTimelapseController = availableControllers.includes(
      "TimelapseController",
    );
    if (selectedPlugin === "Objective" && !hasObjectiveController) {
      setSelectedPlugin("LiveView");
      return;
    }
    if (selectedPlugin === "ExtendedLEDMatrix" && !hasLEDMatrixController) {
      setSelectedPlugin("LiveView");
      return;
    }
    if (
      selectedPlugin === "DPCController" &&
      (!hasLEDMatrixController || !hasDPCController)
    ) {
      setSelectedPlugin("LiveView");
      return;
    }
    if (selectedPlugin === "Timelapse" && !hasTimelapseController) {
      setSelectedPlugin("LiveView");
    }
  }, [selectedPlugin, availableControllers, setSelectedPlugin]);

  return { availableControllers };
};

export default useBackendControllerCapabilities;
