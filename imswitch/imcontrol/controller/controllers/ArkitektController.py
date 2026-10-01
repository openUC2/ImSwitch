"""Arkitekt: offer this microscope as an app on an Arkitekt server.

An Arkitekt server (https://arkitekt.live, or a local deployment, e.g. on a
NAS) lets notebooks, other apps and workflows call the actions declared in
build_app(). It stores acquired images in its mikro data service and shows
the microscope's live state.

ArkitektManager owns the connection (device-code login, unbind). This
controller declares what is offered and serves the HTTP endpoints of the
Arkitekt panel: status, bind, cancel, unbind, settings, activity and uploads.

Units: stage positions and distances in µm, in the ImSwitch user frame (what
PositionerController reports). Exposure is in ms. Actions that move hardware
or acquire are refused while an experiment, a recording or a workflow runs,
and they hold the "microscope" lock, so remote calls run one at a time.
"""
import asyncio
import base64
import collections
import datetime
import enum
import functools
import inspect
import itertools
import re
import threading
import time
from typing import Annotated, Any, Dict, Generator, List, Optional

import numpy as np

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, initLogger
from imswitch.imcontrol.model import configfiletools
from ..basecontrollers import ImConWidgetController

# The action interface's version. The server registers the app under it, and
# the stored login is kept per version: bump it only when actions change
# incompatibly, never with ImSwitch's release number.
ARKITEKT_APP_VERSION = "2.0.0"
MAX_ACTIVITY = 50
MAX_UPLOADS = 12
THUMBNAIL_PX = 192
STATE_PUBLISH_PERIOD_S = 1.0
BUSY_STATES = ("running", "paused", "stopping")
HOMEABLE_AXES = ("X", "Y")  # Z homing drives the objective towards its endstop: not remote


def _choices(name: str, values: List[str]) -> enum.Enum:
    """An Enum the Arkitekt UI shows as a dropdown of this setup's devices.
    Member names are identifiers, values the ImSwitch device names."""
    members, used = {}, set()
    for value in values:
        key = re.sub(r"\W+", "_", str(value)).strip("_").upper() or "DEVICE"
        key = f"_{key}" if key[0].isdigit() else key
        while key in used:
            key += "_"
        used.add(key)
        members[key] = value
    return enum.Enum(name, members)


def _plain(value: Any) -> Any:
    """A call argument as the activity log shows it (no clients, no arrays)."""
    if isinstance(value, enum.Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return None


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class ArkitektController(ImConWidgetController):
    """Declares the microscope's Arkitekt actions and serves the Arkitekt panel."""

    sigArkitektStatus = Signal(dict)
    sigArkitektActivity = Signal(dict)
    sigArkitektUpload = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._manager = getattr(self._master, "arkitektManager", None)
        self._activity = collections.deque(maxlen=MAX_ACTIVITY)
        self._uploads = collections.deque(maxlen=MAX_UPLOADS)
        self._activity_lock = threading.Lock()
        self._activity_ids = itertools.count(1)
        self._offered: List[dict] = []
        self._actions: Dict[str, Any] = {}  # name -> declared function, for local calls
        self._useTriggeredGrab = False
        detectors = self._master.detectorsManager.getAllDeviceNames()
        self.mDetector = self._master.detectorsManager[detectors[0]] if detectors else None
        if self._manager is None:
            self._logger.warning("ArkitektManager unavailable; Arkitekt panel only.")
            return
        if getattr(self._setupInfo, "arkitekt", None) is None:
            self._setupInfo.arkitekt = self._manager.info  # what the panel changes gets saved
        self._manager.set_app_factory(self.build_app, ARKITEKT_APP_VERSION)
        self._manager.add_listener(self.sigArkitektStatus.emit)
        self._manager.auto_connect()

    def closeEvent(self):
        if self._manager is not None:
            self._manager.shutdown()

    # ── HTTP: the Arkitekt panel ────────────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def getArkitektStatus(self) -> dict:
        """Connection and what is offered.

        state: unavailable (package missing) | disabled | unbound | connecting |
        awaiting_login (show userCode and approveUrl) | connected | error.
        Also url, appName, deviceId, boundSince, hasStoredLogin, message,
        the settings (autoConnect, allowInsecureTransport, useMikro), offered
        actions [{name, title, description, moves, images}], the latest
        remote calls (activity) and, when connected, publishedState (the
        live MicroscopeState). Live updates: sigArkitektStatus."""
        if self._manager is None:
            return {"state": "unavailable", "available": False,
                    "message": "Arkitekt is not configured (no ArkitektManager)."}
        with self._activity_lock:
            activity = [dict(entry) for entry in self._activity]
        status = self._manager.status()
        if status.get("state") == "connected":
            status["publishedState"] = self._state_snapshot()  # what Arkitekt shows live
        return {**status, "actions": self._offered or self._planned_actions(),
                "activity": activity[::-1], "uploadCount": len(self._uploads)}

    @APIExport(runOnUIThread=False, requestType="POST")
    def bindArkitekt(self, url: str = "", redeemToken: str = "") -> dict:
        """Bind this microscope to the Arkitekt server at *url* (empty = the
        configured one; saved to the setup file). Returns at once.

        Without a stored login, the state becomes awaiting_login with a userCode
        and an approveUrl: open the link, sign in and approve, and the state
        becomes connected. A redeemToken logs in without a browser (it is not
        saved). Network only: nothing on the microscope moves."""
        if self._manager is None:
            return {"status": "error", "message": "Arkitekt is not configured."}
        url = (url or "").strip()
        if url and url != self._manager.info.url:
            self._manager.info.url = url
            self._save_settings()
        return self._manager.bind(url=url or None, redeem_token=redeemToken or None)

    @APIExport(runOnUIThread=False, requestType="POST")
    def cancelArkitekt(self) -> dict:
        """Stop a pending login, or disconnect. The stored login is kept, so
        the next bind (or the next start, with autoConnect) needs no approval."""
        if self._manager is None:
            return {"status": "error", "message": "Arkitekt is not configured."}
        return self._manager.cancel()

    @APIExport(runOnUIThread=False, requestType="POST")
    def unbindArkitekt(self) -> dict:
        """Disconnect and forget the stored login on this microscope; the next
        bind needs a new approval. The server keeps the app's registration
        (there is no revocation endpoint): remove it in the Arkitekt web UI to
        revoke it there too."""
        if self._manager is None:
            return {"status": "error", "message": "Arkitekt is not configured."}
        return self._manager.unbind()

    @APIExport(runOnUIThread=False, requestType="POST")
    def setArkitektSettings(self, url: Optional[str] = None, appName: Optional[str] = None,
                            autoConnect: Optional[bool] = None,
                            allowInsecureTransport: Optional[bool] = None,
                            useMikro: Optional[bool] = None) -> dict:
        """Change the Arkitekt settings and save them to the setup file. They
        apply at the next bind. A different url or appName is a different
        login (each has its own stored session)."""
        if self._manager is None:
            return {"status": "error", "message": "Arkitekt is not configured."}
        info = self._manager.info
        for key, value in (("url", url), ("appName", appName), ("autoConnect", autoConnect),
                           ("allowInsecureTransport", allowInsecureTransport),
                           ("useMikro", useMikro)):
            if value is not None:
                setattr(info, key, value.strip() if isinstance(value, str) else bool(value))
        saved = self._save_settings()
        status = self._manager.status()
        self.sigArkitektStatus.emit(status)
        return {**status, "status": "success" if saved else "warning",
                "message": None if saved else "Applied, but the setup file could not be saved."}

    @APIExport(runOnUIThread=False)
    def getArkitektUploads(self) -> List[dict]:
        """The latest images sent to Arkitekt, newest first: [{id, name,
        datasetId, shape, dtype, positionUm: {x, y, z}, pixelSizeUm, time,
        thumbnail (JPEG data URL)}]. Live: sigArkitektUpload."""
        return list(self._uploads)[::-1]

    @APIExport(runOnUIThread=False, requestType="POST")
    def clearArkitektActivity(self) -> dict:
        """Forget the activity log and the upload previews (nothing on the server)."""
        with self._activity_lock:
            self._activity.clear()
        self._uploads.clear()
        return {"status": "success"}

    def _save_settings(self) -> bool:
        try:
            configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)
            return True
        except Exception as e:
            self._logger.error(f"Could not save the Arkitekt settings: {e}")
            return False

    # ── what Arkitekt may call ──────────────────────────────────────────────

    def build_app(self, identifier: str, version: str):
        """Declare the microscope as an arkitekt App (on the manager's thread).

        Dropdowns list this setup's devices: positioners, axes and
        illumination sources. Image actions are offered only with mikro
        (setup arkitekt.useMikro), because they store their images there."""
        from arkitekt import App

        mikro_types = self._mikro_types() if self._manager.info.useMikro else None
        app = App(identifier, version,
                  description="openUC2 ImSwitch microscope: stage, illumination, camera",
                  services=[mikro_types["service"]] if mikro_types else [])
        offered: List[dict] = []
        offer = self._offerer(app, offered)
        lasers = list(self._master.lasersManager.getAllDeviceNames())
        illumination = _choices("Illumination", lasers) if lasers else None
        StagePosition = self._declare_state(app)
        if self._master.positionersManager.getAllDeviceNames():
            self._declare_stage_actions(offer, StagePosition)
        if illumination is not None:
            self._declare_illumination_action(offer, illumination)
        if self.mDetector is not None:
            self._declare_camera_action(offer)
            if mikro_types:
                self._declare_image_actions(offer, mikro_types, illumination)
        self._offered = offered
        return app

    def _offerer(self, app, offered: List[dict]):
        """@app.action, plus the activity log and the panel's action list."""
        def offer(moves=False, images=False, **options):
            def declare(fn):
                doc = inspect.getdoc(fn) or fn.__name__
                title, _, description = doc.partition("\n")
                offered.append({"name": fn.__name__, "title": title.strip(),
                                "description": description.strip(), "moves": moves,
                                "images": images})
                action = app.action(**options)(self._tracked(fn))
                self._actions[fn.__name__] = action
                return action
            return declare
        return offer

    def _declare_state(self, app):
        """StagePosition (a returned model) and the live MicroscopeState."""

        # The server only accepts identifiers of the form @package/key; the
        # default would be the bare snake_case class name.
        @app.model(identifier="@imswitch/stage_position")
        class StagePosition:
            """Stage position in µm (ImSwitch user frame)."""
            x: float
            y: float
            z: float
            a: float

        @app.state
        class MicroscopeState:
            """What the microscope is doing, published live."""
            x_um: float = 0.0
            y_um: float = 0.0
            z_um: float = 0.0
            illumination_on: str = ""
            exposure_ms: float = 0.0
            running_action: str = ""

        @app.startup
        def publish_initial_state() -> MicroscopeState:
            return MicroscopeState(**self._state_snapshot())

        @app.background
        async def publish_state(state: MicroscopeState) -> None:
            while True:
                for key, value in self._state_snapshot().items():
                    if getattr(state, key) != value:
                        setattr(state, key, value)
                await asyncio.sleep(STATE_PUBLISH_PERIOD_S)

        return StagePosition

    def _declare_stage_actions(self, offer, StagePosition):
        from arkitekt import Description, Effects, Task

        positioners = list(self._master.positionersManager.getAllDeviceNames())
        Positioner = _choices("Positioner", positioners)
        axes = list(getattr(self._master.positionersManager[positioners[0]], "axes", "XYZ"))
        Axis = _choices("Axis", axes)
        HomeAxis = _choices("HomeAxis", [a for a in axes if a in HOMEABLE_AXES] or ["X"])
        OptionalPositioner = Annotated[Optional[Positioner],
                                       Description("Default: the first positioner")]

        @offer(effects=Effects.NONE)
        def get_stage_position(positioner: OptionalPositioner = None) -> StagePosition:
            """Get Stage Position
            Where the stage is, in µm (ImSwitch user frame), read from the device."""
            return StagePosition(**self._position(positioner, fresh=True))

        @offer(moves=True, effects=Effects.IRREVERSIBLE, locks=["microscope"])
        def move_stage(
            axis: Axis,
            distance_um: Annotated[float, Description(
                "µm to move by; the target position when is_absolute")],
            is_absolute: Annotated[bool, Description(
                "Move to distance_um instead of by it")] = False,
            speed: Annotated[Optional[float], Description(
                "Default: the axis' configured speed")] = None,
            positioner: OptionalPositioner = None,
            *,
            task: Task,
        ) -> StagePosition:
            """Move Stage
            Moves one axis and waits until it arrives. Relative unless
            is_absolute. Z moves the focus: mind the objective."""
            self._refuse_if_busy()
            task.progress(10, f"Moving {axis.value} {'to' if is_absolute else 'by'} "
                              f"{distance_um} µm")
            self._positioner_controller().movePositioner(
                positionerName=self._name(positioner), axis=axis.value, dist=distance_um,
                isAbsolute=is_absolute, isBlocking=True, speed=speed)
            return StagePosition(**self._position(positioner))

        @offer(moves=True, effects=Effects.IRREVERSIBLE, locks=["microscope"])
        def go_to_xy(
            x_um: Annotated[float, Description("Target X, µm")],
            y_um: Annotated[float, Description("Target Y, µm")],
            speed: Annotated[Optional[float], Description(
                "Default: the axes' configured speed")] = None,
            settle_s: Annotated[float, Description("Wait after arriving")] = 0.2,
            positioner: OptionalPositioner = None,
        ) -> StagePosition:
            """Go To XY
            Moves X and Y together to an absolute position and waits.
            Z stays where it is."""
            self._refuse_if_busy()
            self._move_xy(x_um, y_um, speed, positioner)
            time.sleep(max(0.0, settle_s))
            return StagePosition(**self._position(positioner))

        @offer(moves=True, effects=Effects.REPEATABLE, locks=["microscope"])
        def home_axis(axis: HomeAxis, positioner: OptionalPositioner = None) -> StagePosition:
            """Home Axis
            Drives X or Y to its endstop and zeroes it (blocking). Z is
            not homed remotely: use the frame homing in ImSwitch."""
            self._refuse_if_busy()
            self._positioner_controller().homeAxis(
                positionerName=self._name(positioner), axis=axis.value, isBlocking=True)
            return StagePosition(**self._position(positioner))

        @offer(moves=True, effects=Effects.REPEATABLE, locks=["microscope"])
        def move_to_sample_loading_position(
                positioner: OptionalPositioner = None) -> StagePosition:
            """Move To Sample Loading Position
            Drives the stage to the configured loading position (blocking)."""
            self._refuse_if_busy()
            self._positioner_controller().moveToSampleLoadingPosition(
                positionerName=self._name(positioner), is_blocking=True)
            return StagePosition(**self._position(positioner))

    def _declare_illumination_action(self, offer, Illumination):
        from arkitekt import Description, Effects

        @offer(effects=Effects.REPEATABLE, locks=["microscope"])
        def set_illumination(
            channel: Illumination,
            active: bool,
            intensity: Annotated[Optional[float], Description(
                "In the source's own units, within its range; "
                "empty keeps the current setting")] = None,
        ) -> None:
            """Set Illumination
            Switches a light source on or off and optionally sets its
            intensity. Refused outside the source's configured range."""
            self._refuse_if_busy()
            self._set_illumination(channel.value, active, intensity)

    def _declare_camera_action(self, offer):
        from arkitekt import Description, Effects

        @offer(effects=Effects.REPEATABLE)
        def set_camera(
            exposure_ms: Annotated[Optional[float], Description("Empty: unchanged")] = None,
            gain: Annotated[Optional[float], Description("Empty: unchanged")] = None,
        ) -> None:
            """Set Camera
            Sets the exposure time (ms) and/or gain of the camera."""
            self._set_camera(exposure_ms, gain)

    def _declare_image_actions(self, offer, mikro_types, Illumination):
        """acquire_frame and run_tile_scan: images go to mikro (axes c, y, x)."""
        from arkitekt import Description, Effects, Task

        Mikro, Image = mikro_types["Mikro"], mikro_types["Image"]
        IlluminationArg = Annotated[Optional[Illumination] if Illumination else Optional[str],
                                    Description("Light source to switch on during the scan; "
                                                "empty leaves the illumination as it is")]

        @offer(images=True, effects=Effects.REPEATABLE, locks=["microscope"])
        def acquire_frame(
            mikro: Mikro,
            task: Task,
            name: Annotated[Optional[str], Description("Default: frame + time")] = None,
        ) -> Image:
            """Acquire Frame
            Takes one camera frame (captured after the call) and stores it in
            mikro, with the stage position and pixel size in its metadata."""
            self._refuse_if_busy()
            task.progress(20, "Acquiring")
            frame = self.grabCameraFrame(frameSync=2)
            task.progress(60, "Uploading")
            return self._upload(mikro, frame,
                                name or f"Frame {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                                self._position(None))

        @offer(moves=True, images=True, effects=Effects.IRREVERSIBLE, locks=["microscope"])
        def run_tile_scan(
            mikro: Mikro,
            task: Task,
            range_x_um: Annotated[float, Description("Scan width, µm")] = 1000.0,
            range_y_um: Annotated[float, Description("Scan height, µm")] = 1000.0,
            center_x_um: Annotated[Optional[float], Description(
                "Default: the current X")] = None,
            center_y_um: Annotated[Optional[float], Description(
                "Default: the current Y")] = None,
            overlap_percent: Annotated[float, Description(
                "Overlap of neighbouring tiles, used when no step is given")] = 10.0,
            step_x_um: Annotated[Optional[float], Description(
                "Default: from the field of view and the overlap")] = None,
            step_y_um: Annotated[Optional[float], Description(
                "Default: from the field of view and the overlap")] = None,
            illumination: IlluminationArg = None,
            intensity: Annotated[Optional[float], Description(
                "Intensity for the illumination, in its own units")] = None,
            exposure_ms: Annotated[Optional[float], Description("Empty: unchanged")] = None,
            autofocus: Annotated[bool, Description("Autofocus at every tile (moves Z)")] = False,
            autofocus_range_um: float = 100.0,
            autofocus_step_um: float = 10.0,
            speed: Annotated[Optional[float], Description("Default: configured speed")] = None,
            settle_s: Annotated[float, Description("Wait after each move")] = 0.2,
        ) -> Generator[Image, None, None]:
            """Run Tile Scan
            Scans a grid around a centre (snake order), yields each tile as it
            is stored, and places all tiles in one stage space in mikro so the
            Arkitekt viewer shows them stitched by position. Returns the stage
            to where it started and restores the illumination."""
            self._refuse_if_busy()
            if exposure_ms is not None:
                self._set_camera(exposure_ms, None)
            grid = self._tile_grid(range_x_um, range_y_um, center_x_um, center_y_um,
                                   overlap_percent, step_x_um, step_y_um)
            yield from self._tile_scan(
                mikro, task, grid, getattr(illumination, "value", illumination), intensity,
                (autofocus_range_um, autofocus_step_um) if autofocus else None, speed, settle_s)

    def _mikro_types(self) -> Optional[dict]:
        try:
            from mikro import Mikro, mikro_service
            from mikro.arkitekt.specs import MultichannelImage
        except ImportError as e:
            self._logger.warning(f"mikro not installed, no image actions: {e}")
            return None
        return {"Mikro": Mikro, "service": mikro_service, "Image": MultichannelImage}

    # ── activity log (what the panel shows) ─────────────────────────────────

    def _tracked(self, fn):
        """Log each remote call (start, progress, result, error) for the panel.
        functools.wraps keeps the signature the action is declared from; a
        generator stays a generator, so it still streams."""
        name = fn.__name__

        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def generator(*args, **kwargs):
                entry = self._activity_start(name, kwargs)
                count = 0
                try:
                    for item in fn(*args, **kwargs):
                        count += 1
                        self._activity_update(entry, results=count)
                        yield item
                except BaseException as e:
                    self._activity_end(entry, e)
                    raise
                self._activity_end(entry)
            return generator

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            entry = self._activity_start(name, kwargs)
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                self._activity_end(entry, e)
                raise
            self._activity_end(entry)
            return result
        return wrapper

    def _activity_start(self, name: str, kwargs: dict) -> dict:
        arguments = {k: _plain(v) for k, v in kwargs.items()
                     if k not in ("task", "mikro", "state") and _plain(v) is not None}
        entry = {"id": next(self._activity_ids), "action": name, "arguments": arguments,
                 "status": "running", "startedAt": _now(), "started": time.time(),
                 "durationS": None, "progress": None, "results": 0, "error": None}
        with self._activity_lock:
            self._activity.append(entry)
        self.sigArkitektActivity.emit(dict(entry))
        return entry

    def _activity_update(self, entry: dict, **changes) -> None:
        with self._activity_lock:
            entry.update(changes)
        self.sigArkitektActivity.emit(dict(entry))

    def _activity_end(self, entry: dict, error: Optional[BaseException] = None) -> None:
        cancelled = isinstance(error, (GeneratorExit, asyncio.CancelledError))
        self._activity_update(
            entry, status="cancelled" if cancelled else "failed" if error else "done",
            error=None if error is None or cancelled else f"{type(error).__name__}: {error}",
            durationS=round(time.time() - entry["started"], 2))

    def _running_action(self) -> str:
        with self._activity_lock:
            return next((e["action"] for e in reversed(self._activity)
                         if e["status"] == "running"), "")

    # ── hardware helpers ────────────────────────────────────────────────────

    def _refuse_if_busy(self) -> None:
        """Remote calls must not interfere with what runs locally."""
        get = getattr(self._master, "getController", lambda name: None)
        reasons = []
        experiment, recording = get("Experiment"), get("Recording")
        try:
            if experiment is not None and (
                    experiment.getExperimentStatus().get("status") in BUSY_STATES):
                reasons.append("an experiment is running")
            if recording is not None and recording.isRecording():
                reasons.append("a recording is running")
            for name in ("Workflow", "Timelapse"):
                manager = getattr(get(name), "workflow_manager", None)
                if manager is not None and manager.get_status().get("status") in BUSY_STATES:
                    reasons.append(f"a {name.lower()} is running")
        except Exception as e:
            self._logger.warning(f"Arkitekt busy check failed (not blocking): {e}")
        if reasons:
            raise RuntimeError(f"Refused: {', '.join(reasons)} in ImSwitch.")

    def _positioner_controller(self):
        controller = self._master.getController("Positioner")
        if controller is None:
            raise RuntimeError("No PositionerController in this setup.")
        return controller

    def _name(self, positioner) -> str:
        names = self._master.positionersManager.getAllDeviceNames()
        name = getattr(positioner, "value", positioner)
        return name if name in names else names[0]

    def _position(self, positioner, fresh: bool = False) -> Dict[str, float]:
        """{x, y, z, a} in µm, user frame.

        The stage manager's cached position by default: moves and the device's
        position callback keep it current. *fresh* asks the device, as
        PositionerController does, which is a serial round trip on UC2 stages
        (ESP32StageManager.getPosition), so never for the 1 s state publishing."""
        name = self._name(positioner)
        controller = self._master.getController("Positioner") if fresh else None
        if controller is not None:
            position = controller.getPositionerPositions().get(name, {})
        else:
            position = getattr(self._master.positionersManager[name], "position", None) or {}
        return {axis.lower(): float(position.get(axis, 0.0) or 0.0) for axis in "XYZA"}

    def _move_xy(self, x_um: float, y_um: float, speed: Optional[float], positioner) -> None:
        self._positioner_controller().movePositionerXYZ(
            positionerName=self._name(positioner), x=x_um, y=y_um, isAbsolute=True,
            isBlocking=True, speed=speed)

    def _state_snapshot(self) -> dict:
        """The published MicroscopeState; best effort, never raises."""
        state = {"x_um": 0.0, "y_um": 0.0, "z_um": 0.0, "illumination_on": "",
                 "exposure_ms": 0.0, "running_action": self._running_action()}
        try:
            if self._master.positionersManager.getAllDeviceNames():
                position = self._position(None)
                state.update(x_um=position["x"], y_um=position["y"], z_um=position["z"])
            lasers = self._master.lasersManager
            state["illumination_on"] = ", ".join(
                n for n in lasers.getAllDeviceNames() if getattr(lasers[n], "enabled", False))
            state["exposure_ms"] = self._exposure_ms()
        except Exception as e:
            self._logger.debug(f"Arkitekt state snapshot incomplete: {e}")
        return state

    def _set_illumination(self, name: str, active: bool, intensity: Optional[float]) -> None:
        laser = self._master.lasersManager[name]
        if intensity is not None:
            low, high = laser.valueRangeMin, laser.valueRangeMax
            if not low <= intensity <= high:
                raise ValueError(f"{name}: intensity {intensity} outside its range "
                                 f"{low}..{high}.")
        controller = self._master.getController("Laser")
        if controller is not None:
            if intensity is not None:
                controller.setLaserValue(name, intensity)
            controller.setLaserActive(name, active)
        else:
            if intensity is not None:
                laser.setValue(intensity)
            laser.setEnabled(active)

    def _set_camera(self, exposure_ms: Optional[float], gain: Optional[float]) -> None:
        settings = self._master.getController("Settings")
        name = self._master.detectorsManager.getAllDeviceNames()[0]
        if exposure_ms is not None:
            if exposure_ms <= 0:
                raise ValueError("exposure_ms must be positive.")
            if settings is not None:
                settings.setDetectorExposureTime(name, exposure_ms)
            else:
                self.mDetector.setParameter("exposure", exposure_ms)
        if gain is not None:
            if settings is not None:
                settings.setDetectorGain(name, gain)
            else:
                self.mDetector.setParameter("gain", gain)

    def _exposure_ms(self) -> float:
        try:
            value = self.mDetector.getParameter("exposure") if self.mDetector else None
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _pixel_size_um(self) -> float:
        try:
            return float(self.mDetector.pixelSizeUm[-1]) or 1.0
        except Exception:
            return 1.0

    # Deterministic frame grabs, same duck-typed protocol as ExperimentController.

    def _beginTriggeredAcquisition(self) -> bool:
        """Software-trigger mode, so a frame is taken after each move. False
        when the camera cannot (free-run polling is used instead)."""
        if not hasattr(self.mDetector, "setTriggerSource") or not hasattr(
                self.mDetector, "snapSync"):
            self._useTriggeredGrab = False
            return False
        try:
            self._useTriggeredGrab = bool(self.mDetector.setTriggerSource("software"))
        except Exception as e:
            self._logger.warning(f"Could not enable software trigger: {e}")
            self._useTriggeredGrab = False
        return self._useTriggeredGrab

    def _endTriggeredAcquisition(self) -> None:
        """Back to continuous acquisition; safe to call unconditionally."""
        if not self._useTriggeredGrab:
            return
        self._useTriggeredGrab = False
        try:
            self.mDetector.setTriggerSource("continuous")
        except Exception as e:
            self._logger.warning(f"Could not restore continuous mode: {e}")

    def grabCameraFrame(self, frameSync: int = 2) -> np.ndarray:
        """One frame captured after this call: a software trigger when
        _beginTriggeredAcquisition succeeded, else wait until the free-running
        frame counter advanced by *frameSync* (timeout scaled by exposure)."""
        exposure_s = max(self._exposure_ms(), 1.0) / 1000.0
        if self._useTriggeredGrab:
            return self.mDetector.snapSync(timeout=max(2.0, exposure_s * 4 + 1.0))
        if not getattr(self.mDetector, "_running", True):
            self.mDetector.startAcquisition()
        try:
            if hasattr(self.mDetector, "flushBuffer"):
                self.mDetector.flushBuffer()
        except Exception:
            pass
        timeout = max(1.0, (frameSync + 2) * exposure_s + 0.5)
        start, first, frame = time.time(), None, None
        while True:
            try:
                frame, number = self.mDetector.getLatestFrame(returnFrameNumber=True)
            except TypeError:  # a camera without frame numbers: wait frameSync frames instead
                time.sleep((frameSync + 1) * exposure_s)
                return self.mDetector.getLatestFrame()
            first = number if first is None else first
            if number > first + frameSync:
                return frame
            if time.time() - start > timeout:
                self._logger.warning(f"grabCameraFrame: no new frame after {timeout:.1f} s")
                return frame if frame is not None else self.mDetector.getLatestFrame()
            time.sleep(0.01)

    # ── images to mikro ─────────────────────────────────────────────────────

    def _upload(self, mikro, frame: np.ndarray, name: str, position: Dict[str, float],
                space=None):
        """Store *frame* as an array dataset (axes c, y, x, with a pyramid and
        per-channel contrast), register it in *space* at its stage position,
        remember a thumbnail for the panel and return its lens."""
        import xarray as xr
        from mikro import dataset_arrays
        from mikro.api.schema import CoordinateAnchorInput

        array = np.asarray(frame)
        data = array[np.newaxis] if array.ndim == 2 else np.moveaxis(array, -1, 0)
        cyx = xr.DataArray(data, dims=("c", "y", "x"))
        levels = max(1, min(4, int(np.log2(max(cyx.shape[1:]) / 512)) + 1))
        level_zero, scales = dataset_arrays(cyx, levels=levels, method="mean")
        dataset = mikro.create_array_dataset(
            data=level_zero, scales=scales, name=name, axes=["c", "y", "x"],
            anchors=CoordinateAnchorInput.histogram_anchors(cyx))
        pixel_size = self._pixel_size_um()
        if space is not None:
            space.register(dataset, scale={"y": pixel_size, "x": pixel_size},
                           x=position["x"], y=position["y"])
        self._remember_upload(array, name, getattr(dataset, "id", None), position, pixel_size)
        return dataset.lens()

    def _remember_upload(self, array: np.ndarray, name: str, dataset_id, position: dict,
                         pixel_size: float) -> None:
        entry = {"id": next(self._activity_ids), "name": name, "datasetId": dataset_id,
                 "shape": list(array.shape), "dtype": str(array.dtype),
                 "positionUm": {k: position[k] for k in ("x", "y", "z")},
                 "pixelSizeUm": pixel_size, "time": _now(), "thumbnail": _thumbnail(array)}
        self._uploads.append(entry)
        self.sigArkitektUpload.emit(entry)

    def _tile_grid(self, range_x_um, range_y_um, center_x_um, center_y_um, overlap_percent,
                   step_x_um, step_y_um) -> dict:
        """The tile positions (µm, snake order) around a centre (default: here)."""
        here = self._position(None)
        center_x = here["x"] if center_x_um is None else center_x_um
        center_y = here["y"] if center_y_um is None else center_y_um
        if step_x_um is None or step_y_um is None:
            fov_y, fov_x = self._field_of_view_um()
            keep = 1.0 - min(max(overlap_percent, 0.0), 90.0) / 100.0
            step_x_um, step_y_um = step_x_um or fov_x * keep, step_y_um or fov_y * keep
        if step_x_um <= 0 or step_y_um <= 0:
            raise ValueError("Tile steps must be positive.")
        nx, ny = int(range_x_um // step_x_um) + 1, int(range_y_um // step_y_um) + 1
        x0, y0 = center_x - (nx - 1) * step_x_um / 2, center_y - (ny - 1) * step_y_um / 2
        tiles = [(ix, iy, x0 + ix * step_x_um, y0 + iy * step_y_um)
                 for iy in range(ny)
                 for ix in (range(nx) if iy % 2 == 0 else reversed(range(nx)))]
        return {"tiles": tiles, "nx": nx, "ny": ny, "step": (step_x_um, step_y_um),
                "start": here}

    def _tile_scan(self, mikro, task, grid: dict, illumination: Optional[str],
                   intensity: Optional[float], autofocus: Optional[tuple], speed, settle_s):
        from kanne.scalars import Unit
        from mikro import space_2d

        autofocus_controller = self._master.getController("Autofocus") if autofocus else None
        if autofocus and autofocus_controller is None:
            raise RuntimeError("Autofocus requested, but this setup has no autofocus.")
        nx, ny, tiles = grid["nx"], grid["ny"], grid["tiles"]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        space = space_2d(mikro, f"Tile scan {stamp} ({nx}x{ny})", unit=Unit("micrometer"))
        restore_light = self._switch_illumination(illumination, intensity)
        task.progress(0, f"{nx} x {ny} tiles, step {grid['step'][0]:.0f} x "
                         f"{grid['step'][1]:.0f} µm")
        if autofocus_controller is None:  # autofocus switches the trigger mode itself
            self._beginTriggeredAcquisition()
        try:
            for index, (ix, iy, x, y) in enumerate(tiles):
                self._move_xy(x, y, speed, None)
                time.sleep(max(0.0, settle_s))
                if autofocus_controller is not None:
                    self._autofocus(autofocus_controller, *autofocus)
                frame = self.grabCameraFrame(frameSync=2)
                position = {**self._position(None), "x": x, "y": y}
                lens = self._upload(mikro, frame, f"Tile {ix:03d}_{iy:03d} x{x:.0f} y{y:.0f}",
                                    position, space)
                task.progress(int(100 * (index + 1) / len(tiles)),
                              f"Tile {index + 1}/{len(tiles)}")
                yield lens
            space.stage(name=f"Tile scan {stamp}")
        finally:
            self._endTriggeredAcquisition()
            restore_light()
            try:
                self._move_xy(grid["start"]["x"], grid["start"]["y"], speed, None)
            except Exception as e:
                self._logger.warning(f"Could not return to the scan start: {e}")

    def _switch_illumination(self, name: Optional[str], intensity: Optional[float]):
        """Switch *name* on; returns the call that puts it back as it was."""
        if not name:
            return lambda: None
        laser = self._master.lasersManager[name]
        was_on, power = getattr(laser, "enabled", False), getattr(laser, "power", None)
        self._set_illumination(name, True, intensity)

        def restore():
            try:
                self._set_illumination(name, bool(was_on), power)
            except Exception as e:
                self._logger.warning(f"Could not restore {name}: {e}")
        return restore

    def _autofocus(self, controller, range_um: float, step_um: float) -> None:
        """Run the software autofocus here and wait for it: autoFocus() only
        starts a thread. Bounded like the experiment's wait."""
        started = controller.autoFocus(rangez=range_um, resolutionz=step_um)
        if isinstance(started, dict) and started.get("status") == "error":
            raise RuntimeError(f"Autofocus: {started.get('message')}")
        thread = getattr(controller, "_AutofocusThead", None)
        if thread is not None and thread.is_alive():
            steps = 2 * range_um / max(step_um, 0.1) + 1
            thread.join(timeout=30.0 + steps * (1.0 + self._exposure_ms() / 1000.0))
            if thread.is_alive():
                raise RuntimeError("Autofocus did not finish in time.")

    def _field_of_view_um(self) -> tuple:
        """(height, width) of the camera image in µm, from the latest frame."""
        frame = self.mDetector.getLatestFrame()
        if frame is None or np.asarray(frame).ndim < 2:
            raise RuntimeError("No camera image to size the tiles: give step_x_um and step_y_um.")
        rows, cols = np.asarray(frame).shape[:2]
        pixel_size = self._pixel_size_um()
        return rows * pixel_size, cols * pixel_size

    def _planned_actions(self) -> List[dict]:
        """Before the first bind, what would be offered (no arkitekt import)."""
        has_stage = bool(self._master.positionersManager.getAllDeviceNames())
        has_lasers = bool(self._master.lasersManager.getAllDeviceNames())
        has_camera = self.mDetector is not None
        images = has_camera and self._manager.info.useMikro
        planned = [("get_stage_position", "Get Stage Position", False, False, has_stage),
                   ("move_stage", "Move Stage", True, False, has_stage),
                   ("go_to_xy", "Go To XY", True, False, has_stage),
                   ("home_axis", "Home Axis", True, False, has_stage),
                   ("move_to_sample_loading_position", "Move To Sample Loading Position",
                    True, False, has_stage),
                   ("set_illumination", "Set Illumination", False, False, has_lasers),
                   ("set_camera", "Set Camera", False, False, has_camera),
                   ("acquire_frame", "Acquire Frame", False, True, images),
                   ("run_tile_scan", "Run Tile Scan", True, True, images and has_stage)]
        return [{"name": n, "title": t, "description": "", "moves": m, "images": i}
                for n, t, m, i, ok in planned if ok]


def _thumbnail(array: np.ndarray) -> Optional[str]:
    """A small JPEG data URL of a frame, contrast-stretched (1-99 %)."""
    try:
        import cv2
        image = np.asarray(array, dtype=np.float32)
        if image.ndim == 3 and image.shape[-1] not in (3, 4):
            image = image[..., 0]
        low, high = np.percentile(image, (1, 99))
        image = np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        scale = THUMBNAIL_PX / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if image.ndim == 3:
            image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode() if ok else None
    except Exception:
        return None
