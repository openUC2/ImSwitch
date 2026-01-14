import React, { useState, useEffect } from "react";
import { useSelector, useDispatch } from "react-redux";
import "./blockly/customblocks";
import * as Blockly from "blockly";
import { javascriptGenerator } from "blockly/javascript";
import { BlocklyWorkspace } from "react-blockly";
import "./blockly/style.css";
import DarkTheme from "@blockly/theme-dark";

import JSONGenerator from "./blockly/jsonGenerator"; // <-- our custom JSON generator

// Import workflow APIs
import apiExperimentControllerUploadWorkflow from "../backendapi/apiExperimentControllerUploadWorkflow";
import apiExperimentControllerStartWorkflow from "../backendapi/apiExperimentControllerStartWorkflow";
import apiExperimentControllerStopWorkflow from "../backendapi/apiExperimentControllerStopWorkflow";
import apiExperimentControllerPauseWorkflow from "../backendapi/apiExperimentControllerPauseWorkflow";
import apiExperimentControllerResumeWorkflow from "../backendapi/apiExperimentControllerResumeWorkflow";
import apiExperimentControllerGetWorkflowStatus from "../backendapi/apiExperimentControllerGetWorkflowStatus";

// Import workflow state actions
import {
  setWorkflowJson,
  setWorkflowUid,
  setWorkflowStatus,
  setWorkflowError,
  clearWorkflowError,
  setIsGeneratingJson,
  setIsUploading,
  setIsStarting,
  setIsStopping,
  setIsPausing,
  setIsResuming,
  getWorkflowState,
} from "../state/slices/WorkflowSlice";

const initialXml = '<xml xmlns="http://www.w3.org/1999/xhtml"></xml>';
const toolbox = {
  kind: "categoryToolbox",
  contents: [
    {
      kind: "category",
      name: "Logic",
      colour: "#5b80a5",
      contents: [
        {
          kind: "block",
          type: "controls_if",
        },
        {
          kind: "block",
          type: "logic_compare",
          fields: {
            OP: "EQ",
          },
        },
        {
          kind: "block",
          type: "logic_operation",
          fields: {
            OP: "AND",
          },
        },
        {
          kind: "block",
          type: "logic_negate",
        },
        {
          kind: "block",
          type: "logic_boolean",
          fields: {
            BOOL: "TRUE",
          },
        },
        {
          kind: "block",
          type: "logic_null",
        },
        {
          kind: "block",
          type: "logic_ternary",
        },
      ],
    },
    {
      kind: "category",
      name: "Loops",
      colour: "#5ba55b",
      contents: [
        {
          kind: "block",
          type: "controls_repeat_ext",
          values: {
            TIMES: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "10",
              },
            },
          },
        },
        {
          kind: "block",
          type: "controls_whileUntil",
          fields: {
            MODE: "WHILE",
          },
        },
        {
          kind: "block",
          type: "controls_for",
          fields: {
            VAR: {
              id: "XECl4GVs-^+[?1V:UI%Z",
              name: "i",
            },
          },
          values: {
            FROM: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "1",
              },
            },
            TO: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "10",
              },
            },
            BY: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "1",
              },
            },
          },
        },
        {
          kind: "block",
          type: "controls_forEach",
          fields: {
            VAR: {
              id: "zbX-3A%GYRcl`Ngw=KCn",
              name: "j",
            },
          },
        },
        {
          kind: "block",
          type: "controls_flow_statements",
          fields: {
            FLOW: "BREAK",
          },
        },
      ],
    },
    {
      kind: "category",
      name: "Math",
      colour: "#5b67a5",
      contents: [
        {
          kind: "block",
          type: "math_number",
          fields: {
            NUM: "0",
          },
        },
        {
          kind: "block",
          type: "math_arithmetic",
          fields: {
            OP: "ADD",
          },
          values: {
            A: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "1",
              },
            },
            B: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "1",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_single",
          fields: {
            OP: "ROOT",
          },
          values: {
            NUM: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "9",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_trig",
          fields: {
            OP: "SIN",
          },
          values: {
            NUM: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "45",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_constant",
          fields: {
            CONSTANT: "PI",
          },
        },
        {
          kind: "block",
          type: "math_number_property",
          fields: {
            PROPERTY: "EVEN",
          },
          values: {
            NUMBER_TO_CHECK: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "0",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_round",
          fields: {
            OP: "ROUND",
          },
          values: {
            NUM: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "3.1",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_on_list",
          mutation: {
            op: "SUM",
          },
          fields: {
            OP: "SUM",
          },
        },
        {
          kind: "block",
          type: "math_modulo",
          values: {
            DIVIDEND: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "64",
              },
            },
            DIVISOR: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "10",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_constrain",
          values: {
            VALUE: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "50",
              },
            },
            LOW: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "1",
              },
            },
            HIGH: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "100",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_random_int",
          values: {
            FROM: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "1",
              },
            },
            TO: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "100",
              },
            },
          },
        },
        {
          kind: "block",
          type: "math_random_float",
        },
      ],
    },
    {
      kind: "category",
      name: "Text",
      colour: "#5ba58c",
      contents: [
        {
          kind: "block",
          type: "text",
          fields: {
            TEXT: "",
          },
        },
        {
          kind: "block",
          type: "text_join",
          mutation: {
            items: "2",
          },
        },
        {
          kind: "block",
          type: "text_append",
          fields: {
            VAR: {
              id: "7j+{(eU@3Nf`G7G@/J%}",
              name: "item",
            },
          },
          values: {
            TEXT: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_length",
          values: {
            VALUE: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "abc",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_isEmpty",
          values: {
            VALUE: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_indexOf",
          fields: {
            END: "FIRST",
          },
          values: {
            VALUE: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: ",Yd_J]Tvz2@pj)r9I|p$",
                  name: "text",
                },
              },
            },
            FIND: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "abc",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_charAt",
          mutation: {
            at: "true",
          },
          fields: {
            WHERE: "FROM_START",
          },
          values: {
            VALUE: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: ",Yd_J]Tvz2@pj)r9I|p$",
                  name: "text",
                },
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_getSubstring",
          mutation: {
            at1: "true",
            at2: "true",
          },
          fields: {
            WHERE1: "FROM_START",
            WHERE2: "FROM_START",
          },
          values: {
            STRING: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: ",Yd_J]Tvz2@pj)r9I|p$",
                  name: "text",
                },
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_changeCase",
          fields: {
            CASE: "UPPERCASE",
          },
          values: {
            TEXT: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "abc",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_trim",
          fields: {
            MODE: "BOTH",
          },
          values: {
            TEXT: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "abc",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_print",
          values: {
            TEXT: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "abc",
              },
            },
          },
        },
        {
          kind: "block",
          type: "text_prompt_ext",
          mutation: {
            type: "TEXT",
          },
          fields: {
            TYPE: "TEXT",
          },
          values: {
            TEXT: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: "abc",
              },
            },
          },
        },
      ],
    },
    {
      kind: "category",
      name: "Lists",
      colour: "#745ba5",
      contents: [
        {
          kind: "block",
          type: "lists_create_with",
          mutation: {
            items: "0",
          },
        },
        {
          kind: "block",
          type: "lists_create_with",
          mutation: {
            items: "3",
          },
        },
        {
          kind: "block",
          type: "lists_repeat",
          values: {
            NUM: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "5",
              },
            },
          },
        },
        {
          kind: "block",
          type: "lists_length",
        },
        {
          kind: "block",
          type: "lists_isEmpty",
        },
        {
          kind: "block",
          type: "lists_indexOf",
          fields: {
            END: "FIRST",
          },
          values: {
            VALUE: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: "/KVKU4..*N9aqt7tc`r)",
                  name: "list",
                },
              },
            },
          },
        },
        {
          kind: "block",
          type: "lists_getIndex",
          mutation: {
            statement: "false",
          },
          fields: {
            MODE: "GET",
            WHERE: "FROM_START",
          },
          values: {
            VALUE: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: "/KVKU4..*N9aqt7tc`r)",
                  name: "list",
                },
              },
            },
          },
        },
        {
          kind: "block",
          type: "lists_setIndex",
          mutation: {
            at: "true",
          },
          fields: {
            MODE: "SET",
            WHERE: "FROM_START",
          },
          values: {
            LIST: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: "/KVKU4..*N9aqt7tc`r)",
                  name: "list",
                },
              },
            },
          },
        },
        {
          kind: "block",
          type: "lists_getSublist",
          mutation: {
            at1: "true",
            at2: "true",
          },
          fields: {
            WHERE1: "FROM_START",
            WHERE2: "FROM_START",
          },
          values: {
            LIST: {
              kind: "block",
              type: "variables_get",
              fields: {
                VAR: {
                  id: "/KVKU4..*N9aqt7tc`r)",
                  name: "list",
                },
              },
            },
          },
        },
        {
          kind: "block",
          type: "lists_split",
          mutation: {
            mode: "SPLIT",
          },
          fields: {
            MODE: "SPLIT",
          },
          values: {
            DELIM: {
              kind: "block",
              type: "text",
              fields: {
                TEXT: ",",
              },
            },
          },
        },
        {
          kind: "block",
          type: "lists_sort",
          fields: {
            TYPE: "NUMERIC",
            DIRECTION: "1",
          },
        },
      ],
    },
    {
      kind: "category",
      name: "Colour",
      colour: "#a5745b",
      contents: [
        {
          kind: "block",
          type: "colour_picker",
          fields: {
            COLOUR: "#ff0000",
          },
        },
        {
          kind: "block",
          type: "colour_random",
        },
        {
          kind: "block",
          type: "colour_rgb",
          values: {
            RED: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "100",
              },
            },
            GREEN: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "50",
              },
            },
            BLUE: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "0",
              },
            },
          },
        },
        {
          kind: "block",
          type: "colour_blend",
          values: {
            COLOUR1: {
              kind: "block",
              type: "colour_picker",
              fields: {
                COLOUR: "#ff0000",
              },
            },
            COLOUR2: {
              kind: "block",
              type: "colour_picker",
              fields: {
                COLOUR: "#3333ff",
              },
            },
            RATIO: {
              kind: "block",
              type: "math_number",
              fields: {
                NUM: "0.5",
              },
            },
          },
        },
      ],
    },
    {
      kind: "sep",
    },
    {
      kind: "category",
      name: "Variables",
      colour: "#a55b80",
      custom: "VARIABLE",
    },
    {
      kind: "category",
      name: "Functions",
      colour: "#995ba5",
      custom: "PROCEDURE",
    },
    {
      kind: "sep",
    },
    {
      kind: "category",
      name: "Workflow",
      colour: "#cf6b58",
      contents: [
        {
          kind: "block",
          type: "set_laser_power_block",
        },
        {
          kind: "block",
          type: "wait_time_block",
        },
        {
          kind: "block",
          type: "acquire_frame_block",
        },
        {
          kind: "block",
          type: "move_stage_block",
        },
        {
          kind: "block",
          type: "set_exposure_time_block",
        },
        {
          kind: "block",
          type: "set_gain_block",
        },
        {
          kind: "block",
          type: "save_image_block",
        },
        {
          kind: "block",
          type: "autofocus_block",
        },
      ],
    },
  ],
};

const BlocklyController = () => {
  // Redux hooks
  const dispatch = useDispatch();
  const workflowState = useSelector(getWorkflowState);

  // Local state
  const [xml, setXml] = useState("");

  // Generate JSON from workspace
  const handleGenerateJson = async () => {
    try {
      dispatch(setIsGeneratingJson(true));
      dispatch(clearWorkflowError());

      const workspace = Blockly.getMainWorkspace();
      const jsonCode = JSONGenerator.workspaceToCode(workspace);
      console.log("Generated workflow JSON:", jsonCode);

      dispatch(setWorkflowJson(jsonCode));
    } catch (error) {
      console.error("Error generating JSON:", error);
      dispatch(setWorkflowError("Failed to generate workflow JSON"));
    } finally {
      dispatch(setIsGeneratingJson(false));
    }
  };

  // Upload workflow to backend
  const handleUploadWorkflow = async () => {
    if (!workflowState.workflowJson) {
      dispatch(
        setWorkflowError(
          "No workflow JSON to upload. Please generate workflow first."
        )
      );
      return;
    }

    try {
      dispatch(setIsUploading(true));
      dispatch(clearWorkflowError());

      const workflowData = JSON.parse(workflowState.workflowJson);
      const response = await apiExperimentControllerUploadWorkflow(
        workflowData
      );
      console.log("Workflow uploaded successfully:", response);

      // Store the UID from the response
      if (response && response.workflow_id) {
        dispatch(setWorkflowUid(response.workflow_id));
      }

      dispatch(setWorkflowStatus("uploaded"));
    } catch (error) {
      console.error("Error uploading workflow:", error);
      dispatch(setWorkflowError("Failed to upload workflow"));
    } finally {
      dispatch(setIsUploading(false));
    }
  };

  // Start workflow execution
  const handleStartWorkflow = async () => {
    if (!workflowState.workflowUid) {
      dispatch(
        setWorkflowError(
          "No workflow UID available. Please upload workflow first."
        )
      );
      return;
    }

    try {
      dispatch(setIsStarting(true));
      dispatch(clearWorkflowError());

      const response = await apiExperimentControllerStartWorkflow(
        workflowState.workflowUid
      );
      console.log("Workflow started:", response);

      dispatch(setWorkflowStatus("running"));
    } catch (error) {
      console.error("Error starting workflow:", error);
      dispatch(setWorkflowError("Failed to start workflow"));
    } finally {
      dispatch(setIsStarting(false));
    }
  };

  // Stop workflow execution
  const handleStopWorkflow = async () => {
    try {
      dispatch(setIsStopping(true));
      dispatch(clearWorkflowError());

      const response = await apiExperimentControllerStopWorkflow();
      console.log("Workflow stopped:", response);

      dispatch(setWorkflowStatus("stopped"));
    } catch (error) {
      console.error("Error stopping workflow:", error);
      dispatch(setWorkflowError("Failed to stop workflow"));
    } finally {
      dispatch(setIsStopping(false));
    }
  };

  // Pause workflow execution
  const handlePauseWorkflow = async () => {
    try {
      dispatch(setIsPausing(true));
      dispatch(clearWorkflowError());

      const response = await apiExperimentControllerPauseWorkflow();
      console.log("Workflow paused:", response);

      dispatch(setWorkflowStatus("paused"));
    } catch (error) {
      console.error("Error pausing workflow:", error);
      dispatch(setWorkflowError("Failed to pause workflow"));
    } finally {
      dispatch(setIsPausing(false));
    }
  };

  // Resume workflow execution
  const handleResumeWorkflow = async () => {
    try {
      dispatch(setIsResuming(true));
      dispatch(clearWorkflowError());

      const response = await apiExperimentControllerResumeWorkflow();
      console.log("Workflow resumed:", response);

      dispatch(setWorkflowStatus("running"));
    } catch (error) {
      console.error("Error resuming workflow:", error);
      dispatch(setWorkflowError("Failed to resume workflow"));
    } finally {
      dispatch(setIsResuming(false));
    }
  };

  // Handle XML changes in workspace
  const handleXmlChange = (xmlValue) => {
    setXml(xmlValue);
  };

  // Auto-generate JSON when workspace changes (optional)
  useEffect(() => {
    // You can enable auto-generation here if desired
    // handleGenerateJson();
  }, [xml]);

  // Render control buttons based on workflow status
  const renderControlButtons = () => {
    const {
      status,
      isUploading,
      isStarting,
      isStopping,
      isPausing,
      isResuming,
    } = workflowState;

    return (
      <div
        style={{
          display: "flex",
          gap: "10px",
          marginBottom: "10px",
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={handleGenerateJson}
          disabled={workflowState.isGeneratingJson}
          style={{
            backgroundColor: "#4CAF50",
            color: "white",
            padding: "8px 16px",
            border: "none",
            borderRadius: "4px",
          }}
        >
          {workflowState.isGeneratingJson
            ? "Generating..."
            : "Generate Workflow JSON"}
        </button>

        <button
          onClick={handleUploadWorkflow}
          disabled={!workflowState.workflowJson || isUploading}
          style={{
            backgroundColor: "#2196F3",
            color: "white",
            padding: "8px 16px",
            border: "none",
            borderRadius: "4px",
          }}
        >
          {isUploading ? "Uploading..." : "Upload Workflow"}
        </button>

        {(status === "idle" || status === "uploaded" || status === "stopped") &&
          workflowState.workflowUid && (
            <button
              onClick={handleStartWorkflow}
              disabled={isStarting}
              style={{
                backgroundColor: "#FF9800",
                color: "white",
                padding: "8px 16px",
                border: "none",
                borderRadius: "4px",
              }}
            >
              {isStarting ? "Starting..." : "Start Workflow"}
            </button>
          )}

        {status === "running" && (
          <React.Fragment>
            <button
              onClick={handlePauseWorkflow}
              disabled={isPausing}
              style={{
                backgroundColor: "#FF5722",
                color: "white",
                padding: "8px 16px",
                border: "none",
                borderRadius: "4px",
              }}
            >
              {isPausing ? "Pausing..." : "Pause Workflow"}
            </button>

            <button
              onClick={handleStopWorkflow}
              disabled={isStopping}
              style={{
                backgroundColor: "#f44336",
                color: "white",
                padding: "8px 16px",
                border: "none",
                borderRadius: "4px",
              }}
            >
              {isStopping ? "Stopping..." : "Stop Workflow"}
            </button>
          </React.Fragment>
        )}

        {status === "paused" && (
          <React.Fragment>
            <button
              onClick={handleResumeWorkflow}
              disabled={isResuming}
              style={{
                backgroundColor: "#4CAF50",
                color: "white",
                padding: "8px 16px",
                border: "none",
                borderRadius: "4px",
              }}
            >
              {isResuming ? "Resuming..." : "Resume Workflow"}
            </button>

            <button
              onClick={handleStopWorkflow}
              disabled={isStopping}
              style={{
                backgroundColor: "#f44336",
                color: "white",
                padding: "8px 16px",
                border: "none",
                borderRadius: "4px",
              }}
            >
              {isStopping ? "Stopping..." : "Stop Workflow"}
            </button>
          </React.Fragment>
        )}
      </div>
    );
  };

  return (
    <div className="container">
      <h1>Create your Workflow</h1>

      {/* Status Display */}
      <div style={{ marginBottom: "10px" }}>
        <strong>Status: </strong>
        <span
          style={{
            color:
              workflowState.status === "running"
                ? "green"
                : workflowState.status === "error"
                ? "red"
                : workflowState.status === "paused"
                ? "orange"
                : "black",
          }}
        >
          {workflowState.status}
        </span>
        {workflowState.error && (
          <div style={{ color: "red", marginTop: "5px" }}>
            Error: {workflowState.error}
          </div>
        )}
      </div>

      {/* Control Buttons */}
      {renderControlButtons()}

      {/* Blockly Workspace */}
      <BlocklyWorkspace
        className="blockly_container"
        initialXml={initialXml}
        toolboxConfiguration={toolbox}
        workspaceConfiguration={{
          grid: {
            spacing: 20,
            length: 3,
            colour: "#ccc",
            snap: true,
          },
          theme: DarkTheme,
        }}
        onXmlChange={handleXmlChange}
      />

      {/* JSON Output */}
      <pre
        style={{
          backgroundColor: "#272822",
          color: "#fff",
          padding: "1rem",
          marginTop: "10px",
        }}
      >
        {workflowState.workflowJson || "No workflow JSON generated yet."}
      </pre>
    </div>
  );
};

export default BlocklyController;
