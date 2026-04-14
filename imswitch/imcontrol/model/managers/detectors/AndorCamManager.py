# managers/andor_cam_manager.py
"""
DetectorManager for Andor SDK2 / SDK3 cameras.

Provides the same feature-set as HikCamManager so that ImSwitch can drive
an Andor camera (e.g. Zyla, iXon) in exactly the same way as a HIK camera.

The SDK version is chosen via the ``sdkVersion`` key in managerProperties
(or inside the ``andorcam`` sub-dict).  Values: ``2`` or ``3``.
Default is ``2`` on Windows (SDK2 is more widely supported) and ``3``
elsewhere.
"""
from __future__ import annotations

import sys
from typing import List

from imswitch.imcommon.model import initLogger
from .DetectorManager import (
    DetectorManager,
    DetectorAction,
    DetectorNumberParameter,
    DetectorListParameter,
    DetectorBooleanParameter,
)



class AndorCamManager(DetectorManager):
    """
    ImSwitch DetectorManager for Andor SDK2 / SDK3 cameras.

    Mirrors the HikCamManager interface so both camera types can be used
    interchangeably from scripts, widgets, and the REST API.

    Manager properties (JSON / YAML):
      - ``cameraListIndex``    int   – SDK camera index (0-based)
      - ``cameraEffPixelsize`` float – physical pixel size in µm
      - ``mockstackpath``      str   – path for the mock stack (optional)
      - ``mocktype``           str   – "normal" | "random" (optional)
      - ``sdkVersion``         int   – 2 or 3 (default: 2 on Windows, 3 elsewhere)
      - ``andorcam``           dict  – key/value camera settings applied at startup
    """

    def __init__(self, detectorInfo, name, **_extra):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo

        # ---- properties from JSON ----------------------------------------
        cam_idx    = detectorInfo.managerProperties.get("cameraListIndex", 0)
        px_um      = detectorInfo.managerProperties.get("cameraEffPixelsize", 1.0)
        andor_dict = detectorInfo.managerProperties.get("andorcam", {})

        # SDK version: read from top-level property, then from andorcam sub-dict,
        # then default to 2 on Windows and 3 elsewhere.
        _default_sdk = 2 if sys.platform == "win32" else 3
        sdk_version  = int(
            detectorInfo.managerProperties.get(
                "sdkVersion",
                andor_dict.get("sdkVersion", _default_sdk),
            )
        )
        self._sdk_version = sdk_version
        andor_dict = {k: v for k, v in andor_dict.items() if k != "sdkVersion"}

        try:
            self._mockstackpath = detectorInfo.managerProperties["mockstackpath"]
        except KeyError:
            self._mockstackpath = None
        try:
            self._mocktype = detectorInfo.managerProperties["mocktype"]
        except KeyError:
            self._mocktype = "normal"

        # ---- open camera -------------------------------------------------
        self._camera = self._get_cam(cam_idx, sdk_version)

        # Apply startup settings from JSON (same pattern as HikCamManager)
        for prop, val in andor_dict.items():
            self._camera.setPropertyValue(prop, val)

        fullShape = (self._camera.SensorWidth, self._camera.SensorHeight)
        model     = self._camera.model
        self._running             = False
        self._adjustingParameters = False

        # Apply full-frame ROI on start
        self.crop(hpos=0, vpos=0, hsize=fullShape[0], vsize=fullShape[1])

        # Read real values from hardware
        try:
            hw_exp = self._camera.get_exposuretime()  # (cur, min, max) us
            initial_exposure = hw_exp[0] / 1000.0 if hw_exp and hw_exp[0] is not None else 10.0
        except Exception:
            initial_exposure = 10.0
        try:
            hw_gain      = self._camera.get_gain()    # (cur, min, max)
            initial_gain = hw_gain[0] if hw_gain and hw_gain[0] is not None else 0.0
        except Exception:
            initial_gain = 0.0

        # ---- GUI parameters -----------------------------------------------
        parameters = {
            "exposure": DetectorNumberParameter(
                group="Misc",
                value=initial_exposure,
                valueUnits="ms",
                editable=True,
            ),
            "gain": DetectorNumberParameter(
                group="Misc",
                value=initial_gain,
                valueUnits="arb.u.",
                editable=True,
            ),
            "blacklevel": DetectorNumberParameter(
                group="Misc", value=0, valueUnits="arb.u.", editable=True
            ),
            "image_width": DetectorNumberParameter(
                group="Misc", value=fullShape[0], valueUnits="px", editable=False
            ),
            "image_height": DetectorNumberParameter(
                group="Misc", value=fullShape[1], valueUnits="px", editable=False
            ),
            "frame_rate": DetectorNumberParameter(
                group="Misc", value=-1, valueUnits="fps", editable=True
            ),
            "frame_number": DetectorNumberParameter(
                group="Misc", value=0, valueUnits="frames", editable=False
            ),
            "exposure_mode": DetectorListParameter(
                group="Misc",
                value="manual",
                options=["manual", "auto", "single"],
                editable=True,
            ),
            "flat_fielding": DetectorBooleanParameter(
                group="Misc", value=False, editable=True
            ),
            "mode": DetectorBooleanParameter(
                group="Misc", value=name, editable=False
            ),
            "previewMinValue": DetectorNumberParameter(
                group="Misc", value=0, valueUnits="arb.u.", editable=True
            ),
            "previewMaxValue": DetectorNumberParameter(
                group="Misc",
                value=self._getPreviewMaxValue(),
                valueUnits="arb.u.",
                editable=True,
            ),
            "trigger_source": DetectorListParameter(
                group="Acquisition mode",
                value="Continuous",
                options=["Continuous", "Software Trigger", "External Trigger"],
                editable=True,
            ),
            "Camera pixel size": DetectorNumberParameter(
                group="Miscellaneous", value=px_um, valueUnits="um", editable=True
            ),
        }

        # ---- GUI actions -------------------------------------------------
        actions = {
            "More properties": DetectorAction(
                group="Misc", func=self._camera.openPropertiesGUI
            ),
        }

        super().__init__(
            detectorInfo,
            name,
            fullShape=fullShape,
            supportedBinnings=[1],
            model=model,
            parameters=parameters,
            actions=actions,
            croppable=True,
        )


    # -----------------------------------------------------------------------
    # Preview helpers
    # -----------------------------------------------------------------------
    def _getPreviewMaxValue(self) -> int:
        """Return max display value based on pixel encoding bit-depth."""
        try:
            enc = getattr(self._camera, "_activePixelFormat", "Mono16").lower()
            if "16" in enc:
                return 65535
            if "12" in enc:
                return 4095
            if "8" in enc:
                return 255
        except Exception:
            pass
        return 65535  # Safe default for Andor (Mono16)

    # -----------------------------------------------------------------------
    # Parameter interface
    # -----------------------------------------------------------------------
    def setParameter(self, name, value):
        """Set a named parameter and forward it to the camera."""
        super().setParameter(name, value)

        if name not in self._DetectorManager__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')

        value = self._camera.setPropertyValue(name, value)
        return value

    def getParameter(self, name):
        """Read a named parameter from the camera."""
        if name not in self._parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')
        return self._camera.getPropertyValue(name)

    # -----------------------------------------------------------------------
    # Flatfield
    # -----------------------------------------------------------------------
    def setFlatfieldImage(self, flatfieldImage, isFlatfielding: bool):
        self._camera.setFlatfieldImage(flatfieldImage, isFlatfielding)

    def recordFlatfieldImage(self):
        """Average several frames and store as the flatfield reference."""
        self._camera.recordFlatfieldImage()

    # -----------------------------------------------------------------------
    # Frame access
    # -----------------------------------------------------------------------
    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        return self._camera.getLast(returnFrameNumber=returnFrameNumber)

    def getChunk(self):
        try:
            return self._camera.getLastChunk()
        except Exception:
            return None

    def flushBuffers(self):
        self._camera.flushBuffer()

    # -----------------------------------------------------------------------
    # Trigger
    # -----------------------------------------------------------------------
    def setTriggerSource(self, source: str):
        self._performSafeCameraAction(lambda: self._camera.setTriggerSource(source))
        self.parameters["trigger_source"].value = source

    def sendSoftwareTrigger(self):
        """Send a software trigger to the camera."""
        if self._camera.send_trigger():
            self.__logger.debug("Software trigger sent successfully.")
        else:
            self.__logger.warning("Failed to send software trigger.")

    def getCurrentTriggerType(self) -> str:
        return self._camera.getTriggerSource()

    def getTriggerTypes(self) -> List[str]:
        return self._camera.getTriggerTypes()

    # -----------------------------------------------------------------------
    # Acquisition lifecycle
    # -----------------------------------------------------------------------
    def startAcquisition(self):
        if not self._running:
            self._camera.start_live()
            self._running = True
            self.__logger.debug("start_live")

    def stopAcquisition(self):
        if self._running:
            self._running = False
            self._camera.suspend_live()
            self.__logger.debug("suspend_live")

    def stopAcquisitionForROIChange(self):
        self._running = False
        self._camera.stop_live()
        self.__logger.debug("stop_live (for ROI change)")

    # -----------------------------------------------------------------------
    # ROI
    # -----------------------------------------------------------------------
    def crop(self, hpos, vpos, hsize, vsize):
        """
        Crop the sensor readout window.

        hpos  - horizontal start (px)
        vpos  - vertical start (px)
        hsize - width (px)
        vsize - height (px)
        """
        def cropAction():
            self.__logger.debug(
                f"{self._camera.model}: crop to {hsize}x{vsize} at ({hpos},{vpos})"
            )
            self._camera.setROI(hpos, vpos, hsize, vsize)
            self._shape      = (hsize, vsize)
            self._frameStart = (hpos, vpos)

        try:
            self._performSafeCameraAction(cropAction)
        except Exception as e:
            self.__logger.error(e)

    # -----------------------------------------------------------------------
    # Safe action helper (mirrors HikCamManager._performSafeCameraAction)
    # -----------------------------------------------------------------------
    def _performSafeCameraAction(self, function):
        """
        Temporarily stop acquisition, run function, then resume if the stream
        was already running. Prevents SDK errors when changing settings that
        require the camera to be idle.
        """
        self._adjustingParameters = True
        was_running = self._running
        self.stopAcquisitionForROIChange()
        function()
        if was_running:
            self.startAcquisition()
        self._adjustingParameters = False

    # -----------------------------------------------------------------------
    # Pixel size
    # -----------------------------------------------------------------------
    @property
    def pixelSizeUm(self) -> List[float]:
        px = self.parameters["Camera pixel size"].value
        return [1.0, px, px]

    def setPixelSizeUm(self, pixelSizeUm: float):
        self.parameters["Camera pixel size"].value = pixelSizeUm

    # -----------------------------------------------------------------------
    # Housekeeping
    # -----------------------------------------------------------------------
    def finalize(self) -> None:
        super().finalize()
        self.__logger.debug("Safely disconnecting the Andor camera ...")
        self._camera.close()

    def closeEvent(self):
        self._camera.close()

    def openPropertiesDialog(self):
        self._camera.openPropertiesGUI()

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------
    def getCameraStatus(self) -> dict:
        """Return a comprehensive status dict (same structure as HikCamManager)."""
        status = super().getCameraStatus()

        status["cameraType"]            = f"Andor SDK{self._sdk_version}"
        status["isMock"]                = self._mocktype != "normal"
        status["isConnected"]           = self._camera is not None
        status["isAcquiring"]           = self._running
        status["isAdjustingParameters"] = self._adjustingParameters

        try:
            params = self._camera.get_camera_parameters()
            if params:
                status.update(params)
        except Exception as e:
            self.__logger.debug(f"getCameraStatus params: {e}")

        try:
            status["triggerSource"] = self._camera.getTriggerSource()
        except Exception as e:
            self.__logger.debug(f"getCameraStatus trigger: {e}")

        return status

    # -----------------------------------------------------------------------
    # Camera factory
    # -----------------------------------------------------------------------
    def _get_cam(self, cameraId: int, sdk_version: int = 2):
        """Open the Andor camera using the requested SDK version; fall back to TIS mock."""
        try:
            if sdk_version == 2:
                from imswitch.imcontrol.model.interfaces.andorcamera2 import andorcamera2
                self.__logger.debug(
                    f"Trying to initialize Andor SDK2 camera {cameraId}"
                )
                camera = andorcamera2(camera_no=cameraId)
            else:
                from imswitch.imcontrol.model.interfaces.andorcamera import andorcamera
                self.__logger.debug(
                    f"Trying to initialize Andor SDK3 camera {cameraId}"
                )
                camera = andorcamera(camera_no=cameraId)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.warning(
                f"Failed to initialize Andor SDK{sdk_version} camera {cameraId}, "
                "loading TIS mocker"
            )
            from imswitch.imcontrol.model.interfaces.tiscamera_mock import MockCameraTIS
            camera = MockCameraTIS(
                mocktype=self._mocktype,
                mockstackpath=self._mockstackpath,
                isRGB=False,
            )
        self.__logger.info(f"Initialized camera, model: {camera.model}")
        return camera
