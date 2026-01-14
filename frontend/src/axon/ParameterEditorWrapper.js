import ParameterEditorComponent from "./ParameterEditorComponent";
import ExperimentComponent from "./ExperimentComponent";
import { ExperimentDesigner } from "./experiment-designer";

// Feature flag for new experiment designer UI
const USE_NEW_EXPERIMENT_DESIGNER = true;

const ParameterEditorWrapper = () => {
  // Use new dimension-based experiment designer when feature flag is enabled
  if (USE_NEW_EXPERIMENT_DESIGNER) {
    return (
      <div style={{ position: "relative", width: "100%", height: "100%" }}>
        <ExperimentDesigner />
      </div>
    );
  }

  // Legacy layout
  return ( 
    <div style={{ position: "relative", width: "400", height: "300" }}>
        <div style={{ position: "relative", top: "0px", left: "0px", zIndex: 1, }}>
        <ParameterEditorComponent />
        </div>
        <div style={{ }}>
        <ExperimentComponent />
        </div>
    </div>
  );
}

export default ParameterEditorWrapper;

