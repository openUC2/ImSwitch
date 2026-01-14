import * as Blockly from "blockly";
//import { javascriptGenerator, Order } from "blockly/javascript";
import { pythonGenerator, Order } from "blockly/python";

Blockly.defineBlocksWithJsonArray([
  {
    type: "math_number",
    message0: "%1",
    args0: [
      {
        type: "field_number",
        name: "NUM",
        value: 0,
      },
    ],
    output: "Number",
    colour: 230,
    tooltip: "A number.",
    helpUrl: "",
  },
  {
    type: "math_constant",
    message0: "%1",
    args0: [
      {
        type: "field_dropdown",
        name: "CONSTANT",
        options: [
          ["π", "PI"],
          ["e", "E"],
          ["φ", "GOLDEN_RATIO"],
          ["sqrt(2)", "SQRT2"],
          ["sqrt(½)", "SQRT1_2"],
          ["∞", "INFINITY"],
        ],
      },
    ],
    output: "Number",
    colour: 230,
    tooltip: "A mathematical constant.",
    helpUrl: "",
  },
  {
    type: "set_laser_power_block",
    message0: "Set laser power to %1 for channel %2",
    args0: [
      {
        type: "field_number",
        name: "POWER",
        value: 10,
        min: 0,
        max: 1000,
      },
      {
        type: "field_input",
        name: "CHANNEL",
        text: "LED",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 230,
    tooltip: "Sets the laser/LED power on a specific channel",
    helpUrl: "",
  },
  {
    type: "wait_time_block",
    message0: "Wait %1 seconds",
    args0: [
      {
        type: "field_number",
        name: "SECONDS",
        value: 1,
        min: 0,
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 160,
    tooltip: "Wait a certain number of seconds",
    helpUrl: "",
  },
  {
    type: "acquire_frame_block",
    message0: "Acquire frame on channel %1",
    args0: [
      {
        type: "field_input",
        name: "CHANNEL",
        text: "Mono",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 20,
    tooltip: "Acquire an image on the given channel",
    helpUrl: "",
  },
  {
    type: "move_stage_block",
    message0: "Move Stage: X %1 Y %2 Z %3",
    args0: [
      {
        type: "field_variable",
        name: "X",
        value: 0,
      },
      {
        type: "field_variable",
        name: "Y",
        value: 0,
      },
      {
        type: "field_variable",
        name: "Z",
        value: 0,
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 230,
    tooltip: "Move the stage to a specific position",
    helpUrl: "",
  },
  {
    type: "set_exposure_time_block",
    message0: "Set exposure time to %1 ms for channel %2",
    args0: [
      {
        type: "field_number",
        name: "EXPOSURE_TIME",
        value: 100,
        min: 1,
        max: 10000,
      },
      {
        type: "field_input",
        name: "CHANNEL",
        text: "Mono",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 300,
    tooltip: "Sets the camera exposure time for a specific channel",
    helpUrl: "",
  },
  {
    type: "set_gain_block",
    message0: "Set gain to %1 for channel %2",
    args0: [
      {
        type: "field_number",
        name: "GAIN",
        value: 1,
        min: 1,
        max: 100,
      },
      {
        type: "field_input",
        name: "CHANNEL",
        text: "Mono",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 300,
    tooltip: "Sets the camera gain for a specific channel",
    helpUrl: "",
  },
  {
    type: "save_image_block",
    message0: "Save image with filename %1",
    args0: [
      {
        type: "field_input",
        name: "FILENAME",
        text: "image_{timestamp}",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 60,
    tooltip: "Save the acquired image with a specific filename",
    helpUrl: "",
  },
  {
    type: "autofocus_block",
    message0: "Perform autofocus on channel %1",
    args0: [
      {
        type: "field_input",
        name: "CHANNEL",
        text: "Mono",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 180,
    tooltip: "Perform autofocus operation on the specified channel",
    helpUrl: "",
  },
  {
    type: "controls_repeat_ext",
    message0: "repeat %1 times",
    args0: [
      {
        type: "input_value",
        name: "TIMES",
        check: "Number",
      },
    ],
    message1: "do %1",
    args1: [
      {
        type: "input_statement",
        name: "DO",
      },
    ],
    previousStatement: null,
    nextStatement: null,
    colour: 120,
    tooltip: "Repeat a number of times.",
    helpUrl: "",
  },
]);
