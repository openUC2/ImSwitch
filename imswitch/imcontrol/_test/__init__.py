from imswitch.imcontrol.model import Options
from imswitch.imcontrol.view import ViewSetupInfo
import json
optionsBasic = Options.from_json("""
{
    "setupFileName": "",
    "recording": {
        "outputFolder": "D:\\\\Data\\\\",
        "includeDateInOutputFolder": true
    }
}
""")

setupInfoBasic = ViewSetupInfo.from_json("""
{
    "positioners": {
        "VirtualStage": {
          "managerName": "VirtualStageManager",
          "managerProperties": {
            "rs232device": "VirtualMicroscope",
            "isEnable": true,
            "enableauto": false,
            "isDualaxis": 0,
            "stepsizeX": 1,
            "stepsizeY": 1,
            "stepsizeZ": 1,
            "stepsizeA": 1,
            "homeSpeedX": 15000,
            "homeSpeedY": 15000,
            "homeSpeedZ": 15000,
            "homeSpeedA": 15000,
            "homeDirectionX": 1,
            "homeDirectionY": 1,
            "homeDirectionZ": -1,
            "initialSpeed": {"X": 15000, "Y":  15000,"Z": 15000, "A": 15000}
          },
          "axes": [
            "X",
            "Y",
            "Z",
            "A"
          ],
          "forScanning": true,
          "forPositioning": true
        }
      },
    "rs232devices": {
    "VirtualMicroscope": {
      "managerName": "VirtualMicroscopeManager",
      "managerProperties": {
      }
    }
  },
 "lasers": {
    "LED": {
      "analogChannel": null,
      "digitalLine": null,
      "managerName": "VirtualLaserManager",
      "managerProperties": {
        "rs232device": "VirtualMicroscope",
        "channel_index": 1
      },
      "wavelength": 635,
      "valueRangeMin": 0,
      "valueRangeMax": 1023
    }
  },
  "detectors": {
    "WidefieldCamera": {
        "analogChannel": null,
        "digitalLine": null,
        "managerName": "VirtualCameraManager",
        "managerProperties": {
            "isRGB": 0,
            "cameraListIndex": 0,
            "cameraEffPixelsize": 0.2257,
            "virtcam": {
                "exposure": 0,
                "gain": 0,
                "blacklevel": 100,
                "image_width": 400,
                "image_height": 300
            }
        },
        "forAcquisition": true,
        "forFocusLock": true
    }
    },
  "rois": {
    "Full chip": {
      "x": 600,
      "y": 600,
      "w": 1200,
      "h": 1200
    }
  },
  "fovLock": {
    "camera": "WidefieldCamera",
    "positioner": "VirtualStage",
    "updateFreq": 1,
    "piKp":1,
    "piKi":1
    },
    "sim": {
      "monitorIdx": 2,
      "width": 1080,
      "height": 1920,
      "wavelength": 0,
      "pixelSize": 0,
      "angleMount": 0,
      "patternsDir": "/users/bene/ImSwitchConfig/imcontrol_sim/488"
    },
    "dpc": {
      "wavelength": 0.53,
      "pixelsize": 0.2,
      "NA": 0.3,
      "NAi": 0.3,
      "n": 1.0,
      "rotations": [0, 180, 90, 270]
    },
  "PixelCalibration": {},
  "availableWidgets": [
    "Settings",
    "View",
    "Recording",
    "Image",
    "Laser",
    "Positioner",
    "Autofocus",
    "MCT",
    "ROIScan",
    "HistoScan",
    "Hypha"
  ],
  "nonAvailableWidgets":[
    "FocusLock",
    "SIM",
    "DPC",
    "FOVLock",
    "Temperature",
    "HistoScan",
    "PixelCalibration",
    "Lightsheet",
    "WebRTC",
    "Flatfield",
    "STORMRecon",
    "DPC",
    "ImSwitchServer",
    "PixelCalibration",
    "FocusLock"]
}
""", infer_missing=True)

setupInfoWithoutWidgets = ViewSetupInfo.from_json("""
{
  "positioners": {
    "VirtualStage": {
      "managerName": "VirtualStageManager",
      "managerProperties": {
        "rs232device": "VirtualMicroscope",
        "isEnable": true,
        "enableauto": false,
        "isDualaxis": 0,
        "stepsizeX": 1,
        "stepsizeY": 1,
        "stepsizeZ": 1,
        "homeSpeedX": 15000,
        "homeSpeedY": 15000,
        "homeSpeedZ": 15000,
        "homeDirectionX": 1,
        "homeDirectionY": 1,
        "homeDirectionZ": -1,
        "initialSpeed": { "X": 15000, "Y": 15000, "Z": 15000}
      },
      "axes": ["X", "Y", "Z"],
      "forScanning": true,
      "forPositioning": true
    }
  },
  "rs232devices": {
    "VirtualMicroscope": {
      "managerName": "VirtualMicroscopeManager",
      "managerProperties": {
        "imagePath_":"simplant"
      }
    }
  },
  "lasers": {
    "LED": {
      "analogChannel": null,
      "digitalLine": null,
      "managerName": "VirtualLaserManager",
      "managerProperties": {
        "rs232device": "VirtualMicroscope",
        "channel_index": 1
      },
      "wavelength": 635,
      "valueRangeMin": 0,
      "valueRangeMax": 1023
    },
    "LASER": {
      "analogChannel": null,
      "digitalLine": null,
      "managerName": "VirtualLaserManager",
      "managerProperties": {
        "rs232device": "VirtualMicroscope",
        "channel_index": 2
      },
      "wavelength": 488,
      "valueRangeMin": 0,
      "valueRangeMax": 1023
    }
  },
  "detectors": {
    "WidefieldCamera": {
      "analogChannel": null,
      "digitalLine": null,
      "managerName": "VirtualCameraManager",
      "managerProperties": {
        "isRGB": 0,
        "cameraListIndex": 0,
        "cameraEffPixelsize": 1,
        "virtcam": {
          "exposure": 0,
          "gain": 0,
          "blacklevel": 100,
          "image_width": 400,
          "image_height": 300
        }
      },
      "forAcquisition": true,
      "forFocusLock": true
    }
  },
  "rois": {
    "Full chip": {
      "x": 600,
      "y": 600,
      "w": 1200,
      "h": 1200
    }
  },
  "fovLock": {
    "camera": "WidefieldCamera",
    "positioner": "VirtualStage",
    "updateFreq": 1,
    "piKp": 1,
    "piKi": 1
  },
  "objective": {
    "pixelsizes": [0.2, 0.1],
    "NAs": [0.5, 0.8],
    "magnifications": [10, 20],
    "objectiveNames": ["10x", "20x"],
    "objectivePositions": [0, 1],
    "homeDirection": -1,
    "homePolarity": 1,
    "homeSpeed": 20000,
    "homeAcceleration": 20000,
    "calibrateOnStart": true
  },
  "PixelCalibration": {},
  "availableWidgets": [
    "Settings",
    "View",
    "Recording",
    "Image",
    "Laser",
    "Positioner",
    "Autofocus",
    "MCT",
    "ROIScan",
    "HistoScan",
    "Hypha",
    "ImSwitchServer",
    "Lightsheet",
    "Workflow",
    "Lepmon",
    "Experiment",
    "Timelapse",
    "UC2Config",
    "Objective"
  ],
  "nonAvailableWidgets": [
    "Histogramm",
    "imswitch_arkitekt_next",
    "FocusLock",
    "FlowStop",
    "SIM",
        "imswitch_arkitekt",
    "DPC",
    "FOVLock",
    "Temperature",
    "HistoScan",
    "PixelCalibration",
    "WebRTC",
    "Flatfield",
    "STORMRecon",
    "DPC",
    "PixelCalibration",
    "FocusLock"
  ]
}
""", infer_missing=True)

if __name__ == '__main__':
    print(setupInfoBasic)
    print(setupInfoWithoutWidgets)
    print(optionsBasic)

# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
