from imswitch.imcontrol.model import DetectorInfo

detectorInfosBasic = {
    'CAM': DetectorInfo(
        analogChannel=None,
        digitalLine=3,
        managerName='VirtualCameraManager',
        managerProperties={
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
        forAcquisition=True, 
        forFocusLock=False,
    )
}

detectorInfosMulti = {
    'Camera 1': detectorInfosBasic['CAM'],
    'Camera 2': DetectorInfo(
        analogChannel=None,
        digitalLine=5,
        managerName='VirtualCameraManager',
        managerProperties={
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
        forAcquisition=True
    )
}

# TODO: We should probably add a second configuration that differs from the original one - e.g. non squared?
detectorInfosNonSquare = {
    'CAM': detectorInfosBasic['CAM']
}


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
