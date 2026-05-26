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
    if (selectedPlugin === "Objective" && !hasObjectiveController) {
      setSelectedPlugin("LiveView");
    }
  }, [selectedPlugin, availableControllers, setSelectedPlugin]);

  return { availableControllers };
};

export default useBackendControllerCapabilities;
