# managers/andor_cam_manager.py
"""
DetectorManager for Andor SDK3 cameras.

Separates GUI-level logic from low-level SDK control
(see ``interfaces/andor_camera.py``).
"""
from __future__ import annotations

from typing import List

from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model.ImageUtils import rotate_and_flip  # your own helper
from .DetectorManager import (
    DetectorManager,
    DetectorAction,
    DetectorNumberParameter,
    DetectorListParameter,
)



class AndorCamManager(DetectorManager):
    """ImSwitch wrapper around :class:`~interfaces.andor_camera.CameraAndor`."""

    # ------------------------------------------------------------------#
    # Construction                                                      #
    # ------------------------------------------------------------------#
    def __init__(self, detectorInfo, name, **_extra):
        self.__logger = initLogger(self, instanceName=name)
        self.detectorInfo = detectorInfo

        # ---- properties -------------------------------------------------
        cam_idx = detectorInfo.managerProperties.get("cameraListIndex", 0)
        px_um = detectorInfo.managerProperties.get("cameraEffPixelsize", 1)
        andor_props = detectorInfo.managerProperties.get("andor", {})

        # ---- low-level camera ------------------------------------------
        self._camera = self._get_cam(cam_idx)
        for prop, val in andor_props.items():
            if prop == "roi":
                self._camera.setROI(**val)
            elif prop == "gain":
                self._camera.set_gain(val)
            elif prop == "exposure":
                self._camera.set_exposure_time_ms(val)
            elif prop == "trigger_source":
                self._camera.set_trigger_source(val)

        fullShape = (self._camera.SensorWidth, self._camera.SensorHeight)

        # ---- parameters -------------------------------------------------
        parameters = {
            "exposure": DetectorNumberParameter(
                group="Misc", value=self._camera.get_exposure_time_ms(), valueUnits="ms", editable=True
            ),
            "gain": DetectorNumberParameter(
                group="Misc", value=self._camera.get_gain(), valueUnits="arb.u.", editable=True
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
            "trigger_source": DetectorListParameter(
                group="Acquisition mode",
                value="Continuous",
                options=["Continuous", "Software Trigger", "External Trigger"],
                editable=True,
            ),
            "Camera pixel size": DetectorNumberParameter(
                group="Miscellaneous", value=px_um, valueUnits="µm", editable=True
            ),
        }

        actions = {
            "Flush buffers": DetectorAction(group="Misc", func=self.flushBuffers),
        }

        super().__init__(
            detectorInfo,
            name,
            fullShape=fullShape,
            supportedBinnings=[1],
            model="AndorSDK3",
            parameters=parameters,
            actions=actions,
            croppable=False,
        )

        self._running = False

    # ------------------------------------------------------------------#
    # Parameter overrides                                               #
    # ------------------------------------------------------------------#
    def setParameter(self, name, value):
        super().setParameter(name, value)

        if name == "exposure":
            self._camera.set_exposure_time_ms(value)
        elif name == "gain":
            self._camera.set_gain(value)
        elif name == "trigger_source":
            self._camera.set_trigger_source(value)

        return value

    def getParameter(self, name):
        if name == "exposure":
            return self._camera.get_exposure_time_ms()
        if name == "gain":
            return self._camera.get_gain()
        if name == "frame_number":
            return self._camera.frameNumber
        if name == "trigger_source":
            return self._camera.cam.TriggerMode
        return super().getParameter(name)

    # ------------------------------------------------------------------#
    # Acquisition                                                       #
    # ------------------------------------------------------------------#
    def startAcquisition(self):
        if not self._running:
            self._camera.start_live()
            self._running = True

    def stopAcquisition(self):
        if self._running:
            self._camera.stop_live()
            self._running = False

    # GUI helper
    def flushBuffers(self):
        self._camera.flushBuffer()

    # ------------------------------------------------------------------#
    # Frame access                                                      #
    # ------------------------------------------------------------------#
    def getLatestFrame(self, is_resize=True, returnFrameNumber=False):
        frame = self._camera.getLast(returnFrameNumber=returnFrameNumber)
        if isinstance(frame, tuple):
            img, fid = frame
        else:
            img, fid = frame, None

        if img is not None and is_resize:
            img = rotate_and_flip(img)  # project-specific util

        return (img, fid) if returnFrameNumber else img

    # triggers
    def sendSoftwareTrigger(self):
        self._camera.send_trigger()

    # ------------------------------------------------------------------#
    # ImSwitch housekeeping                                             #
    # ------------------------------------------------------------------#
    def finalize(self):
        super().finalize()
        self._camera.close()

    # pixel size metadata
    @property
    def pixelSizeUm(self) -> List[float]:
        px = self.parameters["Camera pixel size"].value
        return [1.0, px, px]

    def setPixelSizeUm(self, pixelSizeUm):
        self.parameters["Camera pixel size"].value = pixelSizeUm

    # ROI convenience
    def crop(self, hpos, vpos, hsize, vsize):
        self._camera.setROI(hpos, vpos, hsize, vsize)


    def _get_cam(self, cameraId, isRGB = False, binning=1):
        try:
            from imswitch.imcontrol.model.interfaces.andorcamera import andorcamera
            self.__logger.debug(f'Trying to initialize Andor camera {cameraId}')
            camera = andorcamera(camera_no=cameraId)
        except Exception as e:
            self.__logger.error(e)
            self.__logger.warning(f'Failed to initialize CameraHik {cameraId}, loading TIS mocker')
            from imswitch.imcontrol.model.interfaces.tiscamera_mock import MockCameraTIS
            camera = MockCameraTIS(mocktype=self._mocktype, mockstackpath=self._mockstackpath,  isRGB=isRGB)
        return camera