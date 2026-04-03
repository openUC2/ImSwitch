"""
SiLA2Controller for OpenUC2 ImSwitch.

Feature classes inherit sila.Feature directly and carry the @sila.* decorators
on their own methods. 
"""

import asyncio
import base64
import io
import json
from typing import Optional

import numpy as np

from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import initLogger, APIExport

try:
    from unitelabs.cdk import sila
    HAS_CDK = True
except ImportError:
    sila = None
    HAS_CDK = False

# ---------------------------------------------------------------------------
# Feature classes
# ---------------------------------------------------------------------------

if HAS_CDK:

    class StageControlFeature(sila.Feature):
        """Provides stage positioning commands for the OpenUC2 microscope."""

        def __init__(self, master, logger):
            super().__init__(
                originator="org.openuc2",
                category="microscopy",
                version="1.0",
            )
            self._master = master
            self._logger = logger

        def _get_positioner(self, name: Optional[str] = None):
            names = self._master.positionersManager.getAllDeviceNames()
            if not names:
                return None
            return self._master.positionersManager[name or names[0]]

        @sila.UnobservableProperty()
        async def get_stage_position(self) -> str:
            """Retrieve the current XYZ position of the stage.

            Returns:
              StagePosition: Comma-separated X,Y,Z coordinates in micrometers.
            """
            p = self._get_positioner()
            if p is None:
                return "0,0,0"
            pos = p.getPosition()
            return f"{pos.get('X', 0)},{pos.get('Y', 0)},{pos.get('Z', 0)}"

        @sila.ObservableProperty()
        async def subscribe_stage_position(self) -> sila.Stream[str]:
            """Stream the current XYZ position of the stage at 2 Hz.

            Returns:
              StagePosition: Comma-separated X,Y,Z coordinates in micrometers.
            """
            while True:
                p = self._get_positioner()
                if p is not None:
                    pos = p.getPosition()
                    yield f"{pos.get('X', 0)},{pos.get('Y', 0)},{pos.get('Z', 0)}"
                else:
                    yield "0,0,0"
                await asyncio.sleep(0.5)

        @sila.UnobservableCommand()
        async def move_stage_to(
            self,
            x_um: float,
            y_um: float,
            z_um: float = 0.0,
            speed: float = 10000.0,
            is_blocking: bool = True,
        ) -> bool:
            """Move the stage to an absolute XYZ position.

            Args:
              x_um: Target X in micrometers.
              y_um: Target Y in micrometers.
              z_um: Target Z in micrometers.
              speed: Movement speed in units per second.
              is_blocking: Wait for movement to complete.

            Returns:
              Result: True if successful.
            """
            p = self._get_positioner()
            if p is None:
                self._logger.error("SiLA2 move_stage_to: no positioner available")
                return False
            try:
                p.move(value=(x_um, y_um), axis="XY", is_absolute=True,
                       is_blocking=is_blocking, speed=(speed, speed))
                if z_um != 0.0:
                    p.move(value=z_um, axis="Z", is_absolute=True,
                           is_blocking=is_blocking, speed=speed)
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 move_stage_to failed: {e}")
                return False

        @sila.UnobservableCommand()
        async def move_stage_relative(
            self,
            dx_um: float = 0.0,
            dy_um: float = 0.0,
            dz_um: float = 0.0,
            speed: float = 10000.0,
            is_blocking: bool = True,
        ) -> bool:
            """Move the stage by a relative offset.

            Args:
              dx_um: Relative X in micrometers.
              dy_um: Relative Y in micrometers.
              dz_um: Relative Z in micrometers.
              speed: Movement speed in units per second.
              is_blocking: Wait for movement to complete.

            Returns:
              Result: True if successful.
            """
            p = self._get_positioner()
            if p is None:
                self._logger.error("SiLA2 move_stage_relative: no positioner available")
                return False
            try:
                if dx_um != 0.0 or dy_um != 0.0:
                    p.move(value=(dx_um, dy_um), axis="XY", is_absolute=False,
                           is_blocking=is_blocking, speed=(speed, speed))
                if dz_um != 0.0:
                    p.move(value=dz_um, axis="Z", is_absolute=False,
                           is_blocking=is_blocking, speed=speed)
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 move_stage_relative failed: {e}")
                return False

        @sila.UnobservableCommand()
        async def home_stage(self, is_blocking: bool = True) -> bool:
            """Home the stage to the loading position.

            Args:
              is_blocking: Wait for homing to complete.

            Returns:
              Result: True if successful.
            """
            p = self._get_positioner()
            if p is None:
                return False
            try:
                if hasattr(p, "moveToSampleLoadingPosition"):
                    p.moveToSampleLoadingPosition(is_blocking=is_blocking)
                elif hasattr(p, "home"):
                    p.home(is_blocking=is_blocking)
                else:
                    self._logger.warning("SiLA2 home_stage: positioner has no home method")
                    return False
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 home_stage failed: {e}")
                return False

    # -----------------------------------------------------------------------

    class ImagingControlFeature(sila.Feature):
        """Provides detector frame acquisition and illumination control."""

        def __init__(self, master, logger):
            super().__init__(
                originator="org.openuc2",
                category="microscopy",
                version="1.0",
            )
            self._master = master
            self._logger = logger

        def _get_detector(self, name: str = ""):
            names = self._master.detectorsManager.getAllDeviceNames()
            if not names:
                return None
            return self._master.detectorsManager[name if name and name in names else names[0]]

        @sila.UnobservableProperty()
        async def get_available_detectors(self) -> str:
            """Get a comma-separated list of available detector names.

            Returns:
              Detectors: Comma-separated detector names.
            """
            return ",".join(self._master.detectorsManager.getAllDeviceNames())

        @sila.UnobservableProperty()
        async def get_available_illumination_sources(self) -> str:
            """Get a comma-separated list of available illumination source names.

            Returns:
              IlluminationSources: Comma-separated illumination source names.
            """
            return ",".join(self._master.lasersManager.getAllDeviceNames())

        @sila.UnobservableCommand()
        async def snap_image(
            self,
            detector_name: str = "",
            exposure_time_ms: float = -1.0,
            gain: float = -1.0,
        ) -> str:
            """Capture a single frame as a base64-encoded PNG.

            Args:
              detector_name: Name of detector (empty = first available).
              exposure_time_ms: Exposure time in ms (-1 = use current).
              gain: Camera gain (-1 = use current).

            Returns:
              FrameBase64: Base64-encoded PNG image data.
            """
            detector = self._get_detector(detector_name)
            if detector is None:
                self._logger.error("SiLA2 snap_image: no detector available")
                return ""
            try:
                if exposure_time_ms > 0:
                    detector.setParameter("exposure", exposure_time_ms)
                if gain >= 0:
                    detector.setParameter("gain", gain)
                frame = detector.getLatestFrame()
                if frame is None:
                    return ""
                from PIL import Image as PILImage
                img = PILImage.fromarray(np.array(frame))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                self._logger.error(f"SiLA2 snap_image failed: {e}")
                return ""

        @sila.UnobservableCommand()
        async def set_illumination(
            self,
            channel_name: str,
            intensity: float,
            enabled: bool = True,
        ) -> bool:
            """Set illumination channel intensity and state.

            Args:
              channel_name: Name of the illumination channel.
              intensity: Intensity value.
              enabled: Whether the channel should be enabled.

            Returns:
              Result: True if successful.
            """
            try:
                laser = self._master.lasersManager[channel_name]
                laser.setValue(intensity)
                laser.setEnabled(1 if enabled else 0)
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 set_illumination failed: {e}")
                return False

        @sila.UnobservableCommand()
        async def set_exposure_time(
            self,
            exposure_time_ms: float,
            detector_name: str = "",
        ) -> bool:
            """Set the exposure time for a detector.

            Args:
              exposure_time_ms: Exposure time in milliseconds.
              detector_name: Name of detector (empty = first available).

            Returns:
              Result: True if successful.
            """
            detector = self._get_detector(detector_name)
            if detector is None:
                return False
            try:
                detector.setParameter("exposure", exposure_time_ms)
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 set_exposure_time failed: {e}")
                return False

    # -----------------------------------------------------------------------

    class ExperimentControlFeature(sila.Feature):
        """Provides commands to submit and monitor automated experiments."""

        def __init__(self, master, logger):
            super().__init__(
                originator="org.openuc2",
                category="microscopy",
                version="1.0",
            )
            self._master = master
            self._logger = logger
            self._experiment_controller = None

        def _get_experiment_controller(self):
            if self._experiment_controller is None:
                self._experiment_controller = self._master.getController("Experiment")
            return self._experiment_controller

        @sila.UnobservableCommand()
        async def start_experiment(self, experiment_json: str) -> str:
            """Start an automated experiment.

            Args:
              experiment_json: JSON string describing the experiment.

            Returns:
              ExperimentId: Identifier for the experiment run.
            """
            ec = self._get_experiment_controller()
            if ec is None:
                return json.dumps({"error": "ExperimentController not available"})
            try:
                from imswitch.imcontrol.controller.controllers.ExperimentController import Experiment
                experiment = Experiment.model_validate_json(experiment_json)
                ec.startWellplateExperiment(experiment)
                return json.dumps({"status": "started", "name": experiment.name})
            except Exception as e:
                self._logger.error(f"SiLA2 start_experiment failed: {e}")
                return json.dumps({"error": str(e)})

        @sila.UnobservableProperty()
        async def get_experiment_status(self) -> str:
            """Get the current status of the running experiment.

            Returns:
              ExperimentStatus: JSON string with status information.
            """
            ec = self._get_experiment_controller()
            if ec is None:
                return json.dumps({"status": "unknown"})
            try:
                status = ec.getExperimentStatus()
                return json.dumps(status) if isinstance(status, dict) else str(status)
            except Exception as e:
                self._logger.error(f"SiLA2 get_experiment_status failed: {e}")
                return json.dumps({"error": str(e)})

        @sila.UnobservableCommand()
        async def pause_experiment(self) -> bool:
            """Pause the currently running experiment.

            Returns:
              Result: True if successful.
            """
            ec = self._get_experiment_controller()
            if ec is None:
                return False
            try:
                ec.pauseExperiment()
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 pause_experiment failed: {e}")
                return False

        @sila.UnobservableCommand()
        async def resume_experiment(self) -> bool:
            """Resume a paused experiment.

            Returns:
              Result: True if successful.
            """
            ec = self._get_experiment_controller()
            if ec is None:
                return False
            try:
                ec.resumeExperiment()
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 resume_experiment failed: {e}")
                return False

        @sila.UnobservableCommand()
        async def stop_experiment(self) -> bool:
            """Stop the currently running experiment.

            Returns:
              Result: True if successful.
            """
            ec = self._get_experiment_controller()
            if ec is None:
                return False
            try:
                ec.stopExperiment()
                return True
            except Exception as e:
                self._logger.error(f"SiLA2 stop_experiment failed: {e}")
                return False

        @sila.UnobservableCommand()
        async def get_experiment_schema(self) -> str:
            """Return the JSON schema for the Experiment data model.

            Returns:
              SchemaJson: JSON schema string for the Experiment model.
            """
            try:
                from imswitch.imcontrol.controller.controllers.ExperimentController import Experiment
                return json.dumps(Experiment.model_json_schema())
            except Exception as e:
                self._logger.error(f"SiLA2 get_experiment_schema failed: {e}")
                return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class SiLA2Controller(ImConWidgetController):
    """Controller for the SiLA2 integration in ImSwitch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._logger.debug("Initializing SiLA2Controller")

        if not HAS_CDK:
            self._logger.warning("unitelabs-cdk not installed; SiLA2 disabled.")
            return
        if not getattr(self._master, "sila2Manager", None):
            self._logger.warning("SiLA2Manager unavailable; controller disabled.")
            return
        if not self._master.sila2Manager.is_enabled():
            self._logger.warning("SiLA2 not enabled; controller disabled.")
            return

        manager = self._master.sila2Manager
        manager.register_feature(StageControlFeature(self._master, self._logger))
        manager.register_feature(ImagingControlFeature(self._master, self._logger))
        manager.register_feature(ExperimentControlFeature(self._master, self._logger))
        manager.start_server()
        self._logger.info("SiLA2Controller initialized – server starting")

    # ------------------------------------------------------------------
    # API-exported helpers (accessible via ImSwitch REST API as well)
    # ------------------------------------------------------------------

    @APIExport(requestType="GET")
    def getSiLA2Status(self) -> dict:
        """Return current SiLA2 server status."""
        mgr = getattr(self._master, "sila2Manager", None)
        if mgr is None:
            return {"enabled": False, "running": False}
        return {
            "enabled": mgr.is_enabled(),
            "running": mgr.is_running(),
            "config": mgr.get_config(),
        }
