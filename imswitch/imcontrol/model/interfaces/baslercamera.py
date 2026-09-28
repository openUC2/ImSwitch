"""Basler camera interface (pypylon, ``pip install pypylon`` - it ships the pylon runtime).

Basler and Allied Vision cameras speak the same GenICam feature language
(SFNC), so :class:`CameraBasler` inherits everything above the SDK from
:class:`~.avcamera.CameraAV` -- hardware ROI, binning, manual/auto/once
exposure, frame-rate cap, software/external trigger with synchronous snaps,
flip, the 3-frame ring buffer and the diagnostics, i.e. the same surface as
:class:`~.hikcamera.CameraHIK` -- and only swaps the SDK plumbing: pypylon node
access, pylon's grab loop and the pixel-format conversion.
"""
import time

from pypylon import genicam, pylon

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.interfaces.avcamera import CameraAV, _isColorFormat

# SFNC 1.x models (ace classic GigE, the pylon camera emulator) only have the
# integer "Raw" variants. ExposureTimeAbs/AcquisitionFrameRateAbs are already
# tried by CameraAV itself.
_FEATURE_ALIASES = {
    'Gain': ('Gain', 'GainRaw'),
    'BlackLevel': ('BlackLevel', 'BlackLevelRaw'),
}


class _Node:
    """VmbPy-style face (get/set/get_range/...) on a pypylon node: the
    interface CameraAV's feature logic is written against."""

    def __init__(self, node):
        self._node = node

    def get(self):
        return self._node.Value

    def set(self, value):
        if isinstance(self._node, genicam.IInteger):
            value = int(round(value))  # SWIG refuses a float for an int64 node
        self._node.Value = value

    def get_range(self):
        return self._node.Min, self._node.Max

    def get_increment(self):
        return self._node.Inc

    def get_available_entries(self):
        return self._node.Symbolics

    def run(self):
        self._node.Execute()


class _GrabHandler(pylon.ImageEventHandler):
    """Runs on pylon's own grab-loop thread, once per frame."""

    def __init__(self, onGrabbed):
        super().__init__()
        self._onGrabbed = onGrabbed

    def OnImageGrabbed(self, camera, grabResult):
        self._onGrabbed(grabResult)


class CameraBasler(CameraAV):
    """Frame grabbing via pylon's grab-loop thread (event handler, no polling)."""

    def __init__(self, cameraNo=None, isRGB=None, binning=1, flipImage=(False, False),
                 frame_rate=-1, pixel_format=None):
        """
        :param cameraNo: Serial number or user-defined name of the camera, or its
                         index in the pylon device list; None = first camera. A
                         serial number that matches no camera falls back to the
                         first one that is free.
        The other parameters are those of CameraAV.
        """
        self.__logger = initLogger(self, tryInheritParent=True)
        self._handler = None
        self._converter = None
        # Frame ids come from here: pylon's ImageNumber restarts with every
        # StartGrabbing (i.e. every trigger change), ours only ever grows.
        self._grabCount = 0
        super().__init__(cameraNo, isRGB=isRGB, binning=binning, flipImage=flipImage,
                         frame_rate=frame_rate, pixel_format=pixel_format)

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    def _open_camera(self, camera_id):
        tlf = pylon.TlFactory.GetInstance()
        devices = tlf.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler cameras found.")
        serials = [d.GetSerialNumber() for d in devices]
        self.__logger.info(f"Basler cameras found (serial numbers): {serials}")

        cid = '0' if camera_id is None else str(camera_id).strip()
        named = [i for i, d in enumerate(devices)
                 if cid in (d.GetSerialNumber(), d.GetUserDefinedName())]
        if named:
            index = named[0]
        elif cid.isdigit() and int(cid) < len(devices):
            index = int(cid)
        elif cid.isdigit():
            # Stale serial number (camera swapped, setup file copied from another
            # rig): take the first free camera instead of dropping to the mock.
            # A name such as "mock" still ends in the mock.
            free = [i for i, d in enumerate(devices) if tlf.IsDeviceAccessible(d)]
            if not free:
                raise RuntimeError(f"No free Basler camera (serial numbers: {serials})")
            index = free[0]
            self.__logger.warning(
                f"No Basler camera matches cameraListIndex {cid}; using serial number "
                f"{serials[index]}. Put that in the setup file to pin the camera.")
        else:
            raise RuntimeError(f"No Basler camera '{cid}' (serial numbers: {serials})")

        self._camera = pylon.InstantCamera(tlf.CreateDevice(devices[index]))
        self._camera.Open()
        self._camera_open = True
        try:
            self.model = devices[index].GetModelName()
            self._trySet('AcquisitionMode', 'Continuous')
            self._readSensorSize()
        except Exception:
            self._cleanup_context_managers()
            raise
        self.__logger.info(f"Opened Basler {self.model}, serial number {serials[index]}")

    def _cleanup_context_managers(self):
        try:
            if self._camera is not None and self._camera.IsOpen():
                self._camera.Close()
        except Exception as e:
            self.__logger.warning(f"Error closing camera: {e}")
        self._camera_open = False

    # ------------------------------------------------------------------
    # Feature access for CameraAV
    # ------------------------------------------------------------------
    def _feature(self, name):
        """The camera's node wrapped for CameraAV, or None if it has none available."""
        if self._camera is None:
            return None
        for alias in _FEATURE_ALIASES.get(name, (name,)):
            try:
                node = getattr(self._camera, alias)
            except Exception:  # pypylon raises for a node the camera does not have
                continue
            if genicam.IsAvailable(node):
                return _Node(node)
        return None

    def _pixelFormats(self):
        node = self._feature('PixelFormat')
        return {name: name for name in node.get_available_entries()} if node else {}

    def _setPixelFormat(self, name):
        self._feature('PixelFormat').set(name)

    def _getPixelFormat(self):
        return str(self._feature('PixelFormat').get())

    # ------------------------------------------------------------------
    # Frame delivery
    # ------------------------------------------------------------------
    def _makeConverter(self):
        """pylon converter for what numpy cannot take as-is: colour -> RGB8,
        bit-packed mono (Mono12p, Mono12Packed) -> Mono16. None otherwise."""
        fmt = str(self._activePixelFormat)
        if _isColorFormat(fmt):
            target = pylon.PixelType_RGB8packed
        elif fmt.endswith('p') or fmt.endswith('Packed'):
            target = pylon.PixelType_Mono16
        else:
            return None
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = target
        # keep 12-bit values at 0..4095 in the 16-bit container (previewMaxValue)
        converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
        return converter

    def _frameToNumpy(self, grabResult):
        if self._converter is not None:
            arr = self._converter.Convert(grabResult).GetArray()
        else:
            arr = grabResult.GetArray()
        return self._finishFrame(arr)

    def _onGrabbed(self, grabResult):
        t_entry = time.time()
        try:
            if not grabResult.GrabSucceeded():
                self._streamStats["incomplete_frames"] += 1
                return
            data = self._frameToNumpy(grabResult)
            self._grabCount += 1
            self._pushFrame(data, self._grabCount, grabResult.TimeStamp, t_entry)
        except Exception as e:
            self.__logger.error(f"Error in grab handler: {e}")

    def start_live(self):
        if self.is_streaming:
            return
        self.flushBuffer()
        self._streamStats = self._newStreamStats()
        self._converter = self._makeConverter()
        if self._handler is None:
            self._handler = _GrabHandler(self._onGrabbed)
            self._camera.RegisterImageEventHandler(
                self._handler, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_None)
        # Only the newest frame is kept, so a slow consumer can never pile up a
        # backlog of stale frames (the growing live-view latency of the HIK).
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly,
                                   pylon.GrabLoop_ProvidedByInstantCamera)
        self.is_streaming = True
        self.__logger.debug("Camera streaming started.")

    def stop_live(self):
        if not self.is_streaming:
            return
        try:
            self._camera.StopGrabbing()
        except Exception as e:
            self.__logger.warning(f"Error stopping camera streaming: {e}")
        self.is_streaming = False
        self.__logger.debug("Camera streaming stopped.")


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
