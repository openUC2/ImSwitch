from .AVManager import AVManager


class BaslerManager(AVManager):
    """ DetectorManager that deals with Basler cameras (pypylon) and the
    parameters for frame extraction from them. Same surface as
    ``HikCamManager``/``AVManager`` -- binning, hardware ROI, auto exposure,
    frame-rate cap, software/external trigger and synchronous snaps -- and the
    same code: Basler and Allied Vision are both GenICam cameras, only the
    camera class (``CameraBasler`` instead of ``CameraAV``) differs.

    Manager properties:

    - ``cameraListIndex`` -- the camera's serial number or user-defined name,
      or its index in the pylon device list (list indexing starts at 0); a
      serial number that matches no camera falls back to the first free one.
      Set this to a name that matches nothing, e.g. "mock", to load a mocker
    - ``isRGB`` -- force colour (1) or mono (0) output; omit to auto-detect
    - ``binning`` -- startup binning factor (default 1)
    - ``supportedBinnings`` -- list of selectable binning factors (default:
      ``[1, 2, 4, 8]`` limited to what the camera reports)
    - ``basler`` -- dictionary of camera properties applied at startup
      (``exposure`` [ms], ``gain``, ``blacklevel``, ``frame_rate`` [fps, -1 =
      no cap], ``pixel_format`` [e.g. "Mono8"], ``exposure_mode``)
    """

    _configKey = 'basler'
    _cameraType = 'Basler'

    def _openCamera(self, cameraId, **kwargs):
        from imswitch.imcontrol.model.interfaces.baslercamera import CameraBasler
        return CameraBasler(cameraId, **kwargs)


# Copyright (C) ImSwitch developers 2021
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
