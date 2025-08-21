'''
#TODO: create experiment controller test based on following api description

First set experiment parameters, then start experiment, get status and stop it 

    "/ExperimentController/forceStopExperiment": {
      "get": {
        "summary": "Forcestopexperiment",
        "description": "Force stop the experiment. Works for both normal and performance modes.",
        "operationId": "forceStopExperiment_ExperimentController_forceStopExperiment_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/getExperimentStatus": {
      "get": {
        "summary": "Getexperimentstatus",
        "description": "Get the current status of running experiments.",
        "operationId": "getExperimentStatus_ExperimentController_getExperimentStatus_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/getHardwareParameters": {
      "get": {
        "summary": "Gethardwareparameters",
        "operationId": "getHardwareParameters_ExperimentController_getHardwareParameters_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/getLastScanAsOMEZARR": {
      "get": {
        "summary": "Getlastscanasomezarr",
        "description": "Returns the last OME-Zarr folder as a zipped file for download.",
        "operationId": "getLastScanAsOMEZARR_ExperimentController_getLastScanAsOMEZARR_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/getOMEWriterConfig": {
      "get": {
        "summary": "Getomewriterconfig",
        "description": "Get current OME writer configuration.",
        "operationId": "getOMEWriterConfig_ExperimentController_getOMEWriterConfig_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/pauseWorkflow": {
      "get": {
        "summary": "Pauseworkflow",
        "description": "Pause the workflow. Only works in normal mode.",
        "operationId": "pauseWorkflow_ExperimentController_pauseWorkflow_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/resumeExperiment": {
      "get": {
        "summary": "Resumeexperiment",
        "description": "Resume the experiment. Only works in normal mode.",
        "operationId": "resumeExperiment_ExperimentController_resumeExperiment_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/startFastStageScanAcquisition": {
      "get": {
        "summary": "Startfaststagescanacquisition",
        "description": "Full workflow: arm camera ➔ launch writer ➔ execute scan.",
        "operationId": "startFastStageScanAcquisition_ExperimentController_startFastStageScanAcquisition_get",
        "parameters": [
          {
            "name": "xstart",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "default": 0, "title": "Xstart" }
          },
          {
            "name": "xstep",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "default": 500, "title": "Xstep" }
          },
          {
            "name": "nx",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "default": 10, "title": "Nx" }
          },
          {
            "name": "ystart",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "default": 0, "title": "Ystart" }
          },
          {
            "name": "ystep",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "default": 500, "title": "Ystep" }
          },
          {
            "name": "ny",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "default": 10, "title": "Ny" }
          },
          {
            "name": "tsettle",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "default": 90, "title": "Tsettle" }
          },
          {
            "name": "tExposure",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "default": 50, "title": "Texposure" }
          },
          {
            "name": "illumination0",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "title": "Illumination0" }
          },
          {
            "name": "illumination1",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "title": "Illumination1" }
          },
          {
            "name": "illumination2",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "title": "Illumination2" }
          },
          {
            "name": "illumination3",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "title": "Illumination3" }
          },
          {
            "name": "led",
            "in": "query",
            "required": false,
            "schema": { "type": "number", "title": "Led" }
          },
          {
            "name": "tPeriod",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "default": 1, "title": "Tperiod" }
          },
          {
            "name": "nTimes",
            "in": "query",
            "required": false,
            "schema": { "type": "integer", "default": 1, "title": "Ntimes" }
          }
        ],
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HTTPValidationError" }
              }
            }
          }
        }
      }
    },
    "/ExperimentController/startFastStageScanAcquisitionFilePath": {
      "get": {
        "summary": "Startfaststagescanacquisitionfilepath",
        "description": "Returns the file path of the last saved fast stage scan.",
        "operationId": "startFastStageScanAcquisitionFilePath_ExperimentController_startFastStageScanAcquisitionFilePath_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {
                  "type": "string",
                  "title": "Response Startfaststagescanacquisitionfilepath Experimentcontroller Startfaststagescanacquisitionfilepath Get"
                }
              }
            }
          }
        }
      }
    },
    "/ExperimentController/startWellplateExperiment": {
      "post": {
        "summary": "Startwellplateexperiment",
        "operationId": "startWellplateExperiment_ExperimentController_startWellplateExperiment_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/Experiment" }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          },
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HTTPValidationError" }
              }
            }
          }
        }
      }
    },
    "/ExperimentController/stopExperiment": {
      "get": {
        "summary": "Stopexperiment",
        "description": "Stop the experiment. Works for both normal and performance modes.",
        "operationId": "stopExperiment_ExperimentController_stopExperiment_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
    "/ExperimentController/stopFastStageScanAcquisition": {
      "get": {
        "summary": "Stopfaststagescanacquisition",
        "description": "Stop the stage scan acquisition and writer thread.",
        "operationId": "stopFastStageScanAcquisition_ExperimentController_stopFastStageScanAcquisition_get",
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": { "application/json": { "schema": {} } }
          }
        }
      }
    },
'''