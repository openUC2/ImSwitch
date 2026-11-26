"""SiLA 2 features of ImSwitch, served by SiLa2Manager (UniteLabs CDK).

Every handler runs its ImSwitch call in a worker thread (``asyncio.to_thread``), so a long move
or exposure does not stall the SiLA server. Handlers go through the ImSwitch controllers (the
same code paths as the HTTP API), so stage travel limits and the "something else is running"
refusals apply to SiLA clients too. Positions are µm in the stage coordinates the ImSwitch UI
shows.

A refusal the client can act on is a defined execution error (StageLimitReached,
InstrumentBusy, UnknownDevice, NotAvailable, PlateNotCalibrated, NotConfirmed, JobFailed,
InvalidCommandSequence); malformed input is a SiLA validation error, checked before an
observable command is accepted. Moves, autofocus, objective changes and new scans are refused
with InstrumentBusy while an acquisition, job or calibration runs.
"""

import asyncio
import dataclasses
import io
import json
import os
import time
import typing
from datetime import datetime

import numpy as np
from fastapi import HTTPException
from unitelabs.cdk import sila
from unitelabs.cdk.features.core.lock_controller import lock_controller as _cdk_lock
from unitelabs.cdk.features.robot.labware_transfer_manipulator_controller import (
    HandoverPosition,
    InvalidCommandSequence,
    PositionIndex,
)

MICROMETER = sila.constraints.Unit(
    label="µm", components=[sila.constraints.UnitComponent(unit="Meter")], factor=1e-6)
MILLISECOND = sila.constraints.Unit(
    label="ms", components=[sila.constraints.UnitComponent(unit="Second")], factor=1e-3)
Micrometer = typing.Annotated[float, MICROMETER]
Millisecond = typing.Annotated[float, MILLISECOND, sila.constraints.MinimalExclusive(0.0)]
JpegImage = typing.Annotated[bytes, sila.constraints.ContentType("image", "jpeg")]
EXPERIMENT_STATES = ("idle", "running", "paused", "stopping")
JOB_STATES = ("idle", "running", "paused", "finished", "aborted", "failed")


# --- errors -----------------------------------------------------------------------------------

class StageLimitReached(Exception):
    """The target lies outside the stage travel limits of this microscope."""


class InstrumentBusy(Exception):
    """Another acquisition, job or calibration drives the microscope."""


class UnknownDevice(Exception):
    """No detector, illumination source, labware or well of that name."""


class NotAvailable(Exception):
    """The loaded setup has no hardware or controller for this command."""


class PlateNotCalibrated(Exception):
    """The plate position is unknown. Centre the stage over a well and call SetPlateOrigin."""


class NotConfirmed(Exception):
    """The job has ask-first steps (homing, objective change); run it with Confirmed set."""


class JobFailed(Exception):
    """The job ended as failed or aborted, or stopped at a question only an operator can answer."""


_lock_feature = None  # the LockController of the running server, see create_features


def lock_active(lock) -> bool:
    """A held lock that has not timed out; a timeout of 0 s means none (SiLA Lock Controller)."""
    return lock is not None and (lock.lifetime.total_seconds() == 0 or not lock.is_expired)


def touch_lock() -> None:
    """A call on the workcell's behalf: a held lock counts from now (the SiLA lock timeout
    runs from the last use). Called from the server's event loop."""
    lock = _lock_feature._lock if _lock_feature is not None else None
    if lock is not None:
        lock.last_usage = datetime.now()


async def _call(fn, *args, conflict=InstrumentBusy, **kwargs):
    """Run a blocking ImSwitch call in a thread; map HTTP refusals onto defined errors."""
    touch_lock()
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except HTTPException as e:
        if e.status_code == 409:
            raise conflict(str(e.detail)) from None
        if e.status_code == 404:
            raise UnknownDevice(str(e.detail)) from None
        if e.status_code == 503:
            raise NotAvailable(str(e.detail)) from None
        if e.status_code == 403:
            raise NotConfirmed(str(e.detail)) from None
        raise RuntimeError(str(e.detail)) from None


def _pick(name: str, names: list, kind: str) -> str:
    """The named device, or the first one when name is empty."""
    if not names:
        raise NotAvailable(f"This setup has no {kind}.")
    if not name:
        return names[0]
    if name not in names:
        raise UnknownDevice(f"No {kind} '{name}'. Available: {', '.join(names)}.")
    return name


class _ImSwitchFeature(sila.Feature):
    """Common base: the master controller and controller lookup."""

    def __init__(self, master, originator="org.openuc2", category="microscopy", **kwargs):
        super().__init__(originator=originator, category=category, version="1.0", **kwargs)
        self._master = master

    def _controller(self, name: str):
        touch_lock()
        controller = self._master.getController(name)
        if controller is None:
            raise NotAvailable(f"The {name} controller is not loaded in this setup.")
        return controller

    def _busy(self) -> list:
        from imswitch.imcontrol.controller.busy import busy_reasons
        return busy_reasons(self._master)

    def _refuse_if_busy(self) -> None:
        reasons = self._busy()
        if reasons:
            raise InstrumentBusy(f"Refused: {', '.join(reasons)}.")

    def _all_light_off(self) -> None:
        lasers = self._master.lasersManager
        for name in lasers.getAllDeviceNames():
            lasers[name].setEnabled(False)

    def _stage(self):
        manager = self._master.positionersManager
        return manager[_pick("", manager.getAllDeviceNames(), "positioner")]

    def _position(self) -> "StagePosition":
        touch_lock()
        pos = self._stage().getPosition()
        return StagePosition(*(round(float(pos.get(ax, 0.0)), 3) for ax in "XYZ"))

    async def _move(self, is_absolute: bool, **axes) -> "StagePosition":
        self._refuse_if_busy()
        positioner = self._controller("Positioner")
        await _call(positioner.movePositionerXYZ, isAbsolute=is_absolute, isBlocking=True,
                    conflict=StageLimitReached, **axes)
        return await asyncio.to_thread(self._position)


# --- stage --------------------------------------------------------------------------------------

@dataclasses.dataclass
class StagePosition(sila.CustomDataType):
    """
    Stage position, in the stage coordinates the ImSwitch UI shows (calibrated origin).

    Attributes:
      X: Position along X.
      Y: Position along Y.
      Z: Focus position.
    """

    x: Micrometer
    y: Micrometer
    z: Micrometer


@dataclasses.dataclass
class Objective(sila.CustomDataType):
    """
    One objective slot.

    Attributes:
      Slot: Slot index, used by MoveToObjective.
      Name: Objective name.
      Magnification: Magnification.
      NumericalAperture: Numerical aperture.
      PixelSize: Pixel size in the sample plane, 0 when not calibrated.
    """

    slot: int
    name: str
    magnification: float
    numerical_aperture: float
    pixel_size: Micrometer


@dataclasses.dataclass
class TravelLimit(sila.CustomDataType):
    """
    The travel range of one axis.

    Attributes:
      Axis: X, Y or Z.
      Minimum: Lowest allowed position.
      Maximum: Highest allowed position.
    """

    axis: str
    minimum: Micrometer
    maximum: Micrometer


def _or(value, default: float) -> float:
    return float(default if value is None else value)


class StageController(_ImSwitchFeature):
    """
    Move the XY stage and the focus (Z) of the microscope.

    Positions are in µm, in the stage coordinates the ImSwitch UI shows. Moves outside the
    travel limits (TravelLimits) are refused with Stage Limit Reached; moves while an
    acquisition, job or calibration runs with Instrument Busy. Moves block until the stage
    has arrived.
    """

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_travel_limits(self) -> list[TravelLimit]:
        """The travel limits per axis; axes without limits are not listed, and a side without
        a limit is reported as plus or minus 1e9 µm."""
        limits = await _call(self._controller("Positioner").getStageTravelLimits)
        out = []
        for axes in limits.values():
            for axis, lim in axes.items():
                if lim.get("enabled") and {lim.get("min"), lim.get("max")} != {None}:
                    out.append(TravelLimit(axis=axis, minimum=_or(lim.get("min"), -1e9),
                                           maximum=_or(lim.get("max"), 1e9)))
            break  # the first positioner, the one the commands move
        return out

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_loading_position(self) -> StagePosition:
        """The sample loading position: where Prepare For Input / Output and
        MoveToLoadingPosition drive the stage so a robot can put or take the plate."""
        return await asyncio.to_thread(self._loading_position)

    def _loading_position(self) -> StagePosition:
        stage, now = self._stage(), self._position()
        device = getattr(stage, "sampleLoadingPositions", None)
        if device is None:
            raise NotAvailable("This stage has no loading position.")
        # stored in device coordinates; shift by the offset between device and UI coordinates
        user = {"X": now.x, "Y": now.y, "Z": now.z}
        return StagePosition(*(round(float(device.get(ax, 0.0)) - stage.getDevicePositionAxis(ax)
                                     + user[ax], 3) for ax in "XYZ"))

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_position(self) -> StagePosition:
        """The current stage position."""
        return await asyncio.to_thread(self._position)

    @sila.ObservableProperty(errors=[NotAvailable])
    async def subscribe_live_position(self) -> sila.Stream[StagePosition]:
        """The stage position, sent twice per second."""
        while True:
            yield await asyncio.to_thread(self._position)
            await asyncio.sleep(0.5)

    @sila.UnobservableCommand(errors=[StageLimitReached, InstrumentBusy, NotAvailable])
    async def move_to(self, x: Micrometer, y: Micrometer) -> StagePosition:
        """
        Move the stage to an absolute XY position. The focus (Z) stays where it is.

        Args:
          X: Target X.
          Y: Target Y.

        Returns:
          Position: The stage position after the move.
        """
        return await self._move(True, x=x, y=y)

    @sila.UnobservableCommand(errors=[StageLimitReached, InstrumentBusy, NotAvailable])
    async def move_focus_to(self, z: Micrometer) -> StagePosition:
        """
        Move the focus (Z) to an absolute position. Driving the objective into the sample
        damages both; prefer Autofocus or small relative moves near the sample.

        Args:
          Z: Target Z.

        Returns:
          Position: The stage position after the move.
        """
        return await self._move(True, z=z)

    @sila.UnobservableCommand(errors=[StageLimitReached, InstrumentBusy, NotAvailable])
    async def move_by(self, x: Micrometer, y: Micrometer, z: Micrometer) -> StagePosition:
        """
        Move the stage by a relative offset. Axes with offset 0 do not move.

        Args:
          X: Offset along X.
          Y: Offset along Y.
          Z: Offset along Z (focus).

        Returns:
          Position: The stage position after the move.
        """
        axes = {k: v for k, v in (("x", x), ("y", y), ("z", z)) if v != 0.0}
        if not axes:
            return await asyncio.to_thread(self._position)
        return await self._move(False, **axes)

    @sila.UnobservableCommand(errors=[InstrumentBusy, NotAvailable])
    async def move_to_loading_position(self) -> StagePosition:
        """
        Move the stage to the sample loading position (LoadingPosition), where a plate is put
        on or taken off the stage.

        Returns:
          Position: The stage position after the move.
        """
        self._refuse_if_busy()
        await _call(self._controller("Positioner").moveToSampleLoadingPosition, is_blocking=True)
        return await asyncio.to_thread(self._position)

    @sila.UnobservableCommand(errors=[NotAvailable])
    async def teach_loading_position(self) -> None:
        """
        Make the current stage position the sample loading position, where a robot puts and
        takes the plate (Prepare For Input / Output drive there). Saved on the microscope.
        """
        await _call(self._controller("Positioner").setSampleLoadingPosition, useCurrent=True)

    @sila.ObservableCommand(errors=[NotAvailable, InstrumentBusy])
    async def home(self) -> StagePosition:
        """
        Home the stage: Z first and lifted, then X and Y (collision-safe frame homing).
        Drives every axis to its end switch.

        Returns:
          Position: The stage position after homing.
        """
        self._refuse_if_busy()
        result = await _call(self._controller("Positioner").startFrameHoming, isBlocking=True)
        if isinstance(result, dict) and not result.get("success", True):
            raise NotAvailable(result.get("error", "Homing is not supported by this stage."))
        return await asyncio.to_thread(self._position)

    @sila.UnobservableCommand(errors=[NotAvailable])
    async def stop(self) -> None:
        """Stop every stage axis immediately."""
        await _call(self._controller("Positioner").stopAllAxes)


# --- imaging -----------------------------------------------------------------------------------

def _to_jpeg(frame: np.ndarray, max_size: int) -> tuple:
    """Min-max scale a frame to 8 bit and encode it as JPEG; returns (width, height, bytes)."""
    from PIL import Image
    data = np.asarray(frame, dtype=np.float32)
    lo, hi = float(np.min(data)), float(np.max(data))
    data = ((data - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(data)).astype(np.uint8)
    img = Image.fromarray(data)
    if max_size > 0:
        img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.convert("L" if img.mode not in ("L", "RGB") else img.mode).save(buf, "JPEG", quality=90)
    return img.size[0], img.size[1], buf.getvalue()


def _data_dir(*parts: str) -> str:
    from imswitch.imcommon.model import dirtools
    path = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


@dataclasses.dataclass
class IlluminationState(sila.CustomDataType):
    """
    One illumination source and its setting.

    Attributes:
      Source: Name, as SetIllumination takes it.
      Enabled: Whether it is on.
      Intensity: Current intensity, in the units of the source.
      MinimumIntensity: Lowest intensity SetIllumination accepts.
      MaximumIntensity: Highest intensity SetIllumination accepts.
      Units: Units of the intensity, e.g. "mW" or "a.u." (arbitrary units, 0 to the maximum).
      Wavelength: Wavelength in nm; 0 for white or unknown.
    """

    source: str
    enabled: bool
    intensity: float
    minimum_intensity: float
    maximum_intensity: float
    units: str
    wavelength: int


@dataclasses.dataclass
class DetectorExposure(sila.CustomDataType):
    """
    The exposure time of one camera.

    Attributes:
      Detector: Camera name.
      ExposureTime: Exposure time.
    """

    detector: str
    exposure_time: typing.Annotated[float, MILLISECOND]


def _exposure_ms(manager) -> float:
    parameter = getattr(manager, "parameters", {}).get("exposure")
    return float(parameter.value) if parameter is not None else 0.0


class ImagingController(_ImSwitchFeature):
    """
    Take images, set exposure and illumination, and focus.

    An empty detector name means the first camera that takes images (Detectors).
    """

    def _cameras(self) -> list:
        return list(self._master.detectorsManager.getAllDeviceNames(lambda d: d.forAcquisition))

    def _detector(self, name: str) -> str:
        return _pick(name, self._cameras(), "detector")

    def _frame(self, detector: str) -> np.ndarray:
        frames = self._controller("Recording").snapNumpy()
        frame = frames.get(detector)
        if frame is None:
            raise NotAvailable(f"No frame from detector '{detector}'.")
        return frame

    @sila.UnobservableProperty()
    async def get_detectors(self) -> list[str]:
        """Names of the cameras that take images."""
        return self._cameras()

    @sila.UnobservableProperty()
    async def get_exposure_times(self) -> list[DetectorExposure]:
        """The exposure time of each camera."""
        manager = self._master.detectorsManager
        return [DetectorExposure(name, _exposure_ms(manager[name])) for name in self._cameras()]

    @sila.UnobservableProperty()
    async def get_illumination(self) -> list[IlluminationState]:
        """Every illumination source: on or off, intensity, its range and units."""
        return await asyncio.to_thread(self._illumination)

    def _illumination(self) -> list:
        lasers = self._master.lasersManager
        out = []
        for name in lasers.getAllDeviceNames():
            laser = lasers[name]
            out.append(IlluminationState(
                source=name, enabled=bool(getattr(laser, "enabled", False)),
                intensity=float(getattr(laser, "power", 0) or 0),
                minimum_intensity=_or(laser.valueRangeMin, 0.0),
                maximum_intensity=_or(laser.valueRangeMax, 1.0),
                units=str(laser.valueUnits or "a.u."), wavelength=int(laser.wavelength or 0)))
        return out

    @sila.UnobservableProperty()
    async def get_illumination_sources(self) -> list[str]:
        """Names of the illumination sources (lasers and LEDs)."""
        return list(self._master.lasersManager.getAllDeviceNames())

    @sila.UnobservableCommand(errors=[UnknownDevice, NotAvailable])
    async def snap_image(
        self, detector: str, max_size: typing.Annotated[int, sila.constraints.MinimalInclusive(0)]
    ) -> tuple[int, int, JpegImage]:
        """
        Take one frame and return it as an 8-bit JPEG preview (min-max scaled). Nothing is saved.

        Args:
          Detector: Detector name; empty for the first detector.
          MaxSize: Longest edge of the returned image in pixels; 0 for full size.

        Returns:
          Width: Width of the returned image in pixels.
          Height: Height of the returned image in pixels.
          Image: The frame as JPEG.
        """
        name = self._detector(detector)
        frame = await _call(self._frame, name)
        return await asyncio.to_thread(_to_jpeg, frame, max_size)

    @sila.UnobservableCommand(errors=[UnknownDevice, NotAvailable])
    async def save_image(self, detector: str, name: str) -> str:
        """
        Take one frame and save it at full bit depth as OME-TIFF in the data folder
        (subfolder SiLA), with pixel size, stage position and objective.

        Args:
          Detector: Detector name; empty for the first detector.
          Name: Name of the image; the file is <timestamp>_<name>.ome.tif.

        Returns:
          FilePath: Path of the saved file on the microscope.
        """
        det = self._detector(detector)
        frame = await _call(self._frame, det)
        return await asyncio.to_thread(self._save, det, frame, name or "snap")

    def _save(self, detector: str, frame: np.ndarray, name: str) -> str:
        from imswitch.imcontrol.model.aufgaben.storage import save_image
        manager = self._master.detectorsManager[detector]
        pixel_size = getattr(manager, "pixelSizeUm", [1, 1, 1])
        stage = self._master.positionersManager
        pos = stage[stage.getAllDeviceNames()[0]].getPosition() if stage.getAllDeviceNames() else {}
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_data_dir("SiLA"), f"{stamp}_{name}.ome.tif")
        save_image(path, np.asarray(frame), pixel_size_um=float(pixel_size[-1]),
                   origin_um=(pos.get("X", 0.0), pos.get("Y", 0.0), pos.get("Z")),
                   name=name, channel_names=[detector], exposure_ms=[_exposure_ms(manager)])
        return path

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_objectives(self) -> list[Objective]:
        """The objective slots: index, name, magnification, NA and pixel size."""
        st = await _call(self._controller("Objective").getstatus)
        columns = zip(st.get("availableObjectivesNames", []),
                      st.get("availableObjectiveMagnifications", []),
                      st.get("availableObjectiveNAs", []),
                      st.get("availableObjectivePixelSizes", []))
        return [Objective(slot=i, name=str(name), magnification=float(mag or 0),
                          numerical_aperture=float(na or 0), pixel_size=float(px or 0))
                for i, (name, mag, na, px) in enumerate(columns)]

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_current_objective(self) -> int:
        """Slot index of the objective in the light path."""
        return int((await _call(self._controller("Objective").getstatus))["currentObjective"])

    @sila.ObservableCommand(errors=[NotAvailable, InstrumentBusy, UnknownDevice])
    async def move_to_objective(
        self, slot: typing.Annotated[int, sila.constraints.MinimalInclusive(0)], *,
        status: sila.Status,
    ) -> int:
        """
        Switch the objective: the turret moves and Z follows the objective's focus offset.
        The turret sweeps under the sample.

        Args:
          Slot: Objective slot index, see Objectives.

        Returns:
          CurrentObjective: Slot index after the move.
        """
        self._refuse_if_busy()
        objective = self._controller("Objective")
        started = await _call(objective.moveToObjective, slot)
        if not started.get("accepted", False):
            reason = started.get("reason", "refused")
            raise (InstrumentBusy if reason == "busy" else UnknownDevice)(
                f"Objective slot {slot}: {reason}.")
        while (state := await _call(objective.getstatus))["isMovingObjective"]:
            await asyncio.sleep(0.3)
        if state.get("lastObjectiveMoveError"):
            raise RuntimeError(state["lastObjectiveMoveError"])
        return int(state["currentObjective"])

    @sila.UnobservableCommand(errors=[UnknownDevice, NotAvailable])
    async def set_exposure_time(self, detector: str, exposure_time: Millisecond) -> None:
        """
        Set the exposure time of a detector.

        Args:
          Detector: Detector name; empty for the first detector.
          ExposureTime: The exposure time.
        """
        await _call(self._controller("Settings").setDetectorExposureTime,
                    detectorName=self._detector(detector), exposureTime=exposure_time)

    @sila.UnobservableCommand(errors=[UnknownDevice, NotAvailable])
    async def set_illumination(self, source: str, intensity: float, enabled: bool) -> None:
        """
        Set the intensity of an illumination source and switch it on or off. The intensity is in
        the units of the source and must lie in its range (setup valueRangeMin/Max).

        Args:
          Source: Illumination source name, see IlluminationSources.
          Intensity: Intensity in the units of the source.
          Enabled: Switch the source on (true) or off (false).
        """
        if not source:
            raise UnknownDevice("Name the illumination source.")
        name = _pick(source, self._master.lasersManager.getAllDeviceNames(), "illumination source")
        laser = self._master.lasersManager[name]
        lo, hi = laser.valueRangeMin, laser.valueRangeMax
        if lo is not None and hi is not None and not lo <= intensity <= hi:
            raise sila.errors.ValidationError(
                f"Intensity {intensity} outside the range {lo}..{hi} of '{name}'.", "Intensity")
        laser_controller = self._controller("Laser")
        await _call(laser_controller.setLaserValue, name, intensity)
        await _call(laser_controller.setLaserActive, name, enabled)

    @sila.UnobservableCommand()
    async def switch_off_illumination(self) -> None:
        """Switch every illumination source off."""
        await asyncio.to_thread(self._all_light_off)

    @sila.ObservableCommand(errors=[NotAvailable, InstrumentBusy])
    async def autofocus(
        self,
        range_: typing.Annotated[float, MICROMETER, sila.constraints.MinimalExclusive(0.0),
                                 sila.constraints.MaximalInclusive(1000.0)],
        step: typing.Annotated[float, MICROMETER, sila.constraints.MinimalExclusive(0.0)],
        *,
        status: sila.Status,
    ) -> Micrometer:
        """
        Find the focus: scan Z by plus and minus Range around the current focus in steps of Step,
        fit the sharpness peak and move there.

        Args:
          Range: Half the scanned Z range.
          Step: Z step of the scan.

        Returns:
          FocusPosition: Z after the autofocus.
        """
        self._refuse_if_busy()
        af = self._controller("Autofocus")
        started = await _call(af.autoFocus, rangez=range_, resolutionz=step)
        if isinstance(started, dict) and started.get("status") == "error":
            raise InstrumentBusy(started.get("message", "Autofocus refused."))
        t0, expected = time.monotonic(), max(2.0, 2 * range_ / step * 0.3)
        while (state := await _call(af.getAutofocusStatus))["isRunning"]:
            status.update(progress=min(0.95, (time.monotonic() - t0) / expected))
            await asyncio.sleep(0.5)
        if state["state"] not in ("finished", "idle"):
            raise RuntimeError(f"Autofocus ended in state '{state['state']}'.")
        return float(state["currentZ"])


# --- well plates and experiments ---------------------------------------------------------------

class ExperimentController(_ImSwitchFeature):
    """
    Image well plates and run ImSwitch experiments.

    The plate position on the stage is taught once per plate type with SetPlateOrigin (stage
    centred over a known well) and saved in the setup; MoveToWell and ImageWells then address
    wells by name, e.g. "B3".
    """

    def __init__(self, master, origins: dict = None, save=None, **kwargs):
        super().__init__(master, **kwargs)
        self._origins = origins if origins is not None else {}  # load name -> [dx, dy] µm
        self._save = save or (lambda: None)

    def _experiment(self):
        return self._controller("Experiment")

    def _definition(self, load_name: str):
        manager = getattr(self._experiment(), "labware_manager", None)
        if manager is None:
            raise NotAvailable("No labware definitions loaded.")
        try:
            return manager.get(load_name)
        except KeyError:
            raise UnknownDevice(f"No labware '{load_name}'. See Labware.") from None

    def _wells(self, labware: str, tokens: list) -> list:
        """Wells for tokens like "A1" or ranges like "A1:B3", in stage coordinates."""
        from imswitch.imcontrol.model.labware import WellSelectionPattern, resolve_pattern
        lab = self._definition(labware)
        if labware not in self._origins:
            raise PlateNotCalibrated(f"No plate origin for '{labware}'. Call SetPlateOrigin.")
        tokens = [t.strip().upper() for t in tokens]
        ranges = [t for t in tokens if ":" in t]
        singles = [t for t in tokens if ":" not in t]
        pattern = WellSelectionPattern(wells=singles or None, ranges=ranges or None)
        try:
            wells = resolve_pattern(lab, pattern)
        except ValueError as e:
            raise UnknownDevice(str(e)) from None
        dx, dy = self._origins[labware]
        return [(w.id, w.x + dx, w.y + dy) for w in wells]

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_labware(self) -> list[str]:
        """Load names of the plate definitions the microscope knows."""
        return [s.get("load_name", "") for s in self._experiment().getLabwareList()]

    @sila.UnobservableProperty()
    async def get_calibrated_labware(self) -> list[str]:
        """Plate types with a taught plate origin, usable in MoveToWell and ImageWells."""
        return sorted(self._origins)

    @sila.UnobservableCommand(errors=[UnknownDevice, NotAvailable])
    async def set_plate_origin(self, labware: str, well: str) -> None:
        """
        Teach where plates of this type sit: the stage is now centred over the given well.
        Saved on the microscope, so it survives a restart. Well names are not case sensitive.

        Args:
          Labware: Load name of the plate, see Labware.
          Well: The well the stage is centred over, e.g. "A1".
        """
        lab = self._definition(labware)
        w = lab.wells.get(well.strip().upper())
        if w is None:
            raise UnknownDevice(f"'{labware}' has no well '{well}'.")
        pos = await asyncio.to_thread(self._position)
        self._origins[labware] = [pos.x - w.x, pos.y - w.y]
        await asyncio.to_thread(self._save)

    @sila.UnobservableCommand(errors=[UnknownDevice, PlateNotCalibrated, StageLimitReached,
                                      InstrumentBusy, NotAvailable])
    async def move_to_well(self, labware: str, well: str) -> StagePosition:
        """
        Centre the stage over a well of a plate whose origin was taught with SetPlateOrigin.

        Args:
          Labware: Load name of the plate, see CalibratedLabware.
          Well: Well name, e.g. "B3".

        Returns:
          Position: The stage position after the move.
        """
        (_, x, y), = self._wells(labware, [well])
        return await self._move(True, x=x, y=y)

    @sila.ObservableCommand(errors=[UnknownDevice, PlateNotCalibrated, InstrumentBusy,
                                    NotAvailable])
    async def image_wells(
        self,
        labware: str,
        wells: typing.Annotated[list[str], sila.constraints.MinimalElementCount(1)],
        illumination: str,
        intensity: float,
        exposure_time: Millisecond,
        autofocus: bool,
        *,
        status: sila.Status,
    ) -> str:
        """
        Take one image in the centre of each well, as an ImSwitch experiment (one OME-Zarr per
        well on the microscope). The illumination is off afterwards and the stage stays at the
        last well. Progress is an estimate.

        Args:
          Labware: Load name of the plate, see CalibratedLabware.
          Wells: Wells or ranges, e.g. ["A1", "B2:B4"].
          Illumination: Illumination source; empty for the first one.
          Intensity: Intensity in the units of the source.
          ExposureTime: Exposure time per image.
          Autofocus: Run the software autofocus at each well.

        Returns:
          DataPath: Folder with the images on the microscope.
        """
        self._refuse_if_busy()
        wells_xy = self._wells(labware, wells)
        source = _pick(illumination, self._master.lasersManager.getAllDeviceNames(),
                       "illumination source")
        experiment = _well_experiment(wells_xy, source, intensity, exposure_time, autofocus,
                                      labware)
        ec = self._experiment()
        await _call(ec.startWellplateExperiment, experiment)
        seconds = len(wells_xy) * (3.0 + exposure_time / 1000 + (15.0 if autofocus else 0.0))
        try:
            await self._wait_for_experiment(ec, status, seconds)
        finally:
            await asyncio.to_thread(self._all_light_off)
        return getattr(ec, "_last_experiment_dir", "")

    async def _wait_for_experiment(self, ec, status, seconds: float) -> None:
        t0 = time.monotonic()
        await asyncio.sleep(0.5)
        while (await _call(ec.getExperimentStatus)).get("status") in EXPERIMENT_STATES[1:]:
            status.update(progress=min(0.95, (time.monotonic() - t0) / max(seconds, 1.0)))
            await asyncio.sleep(0.5)

    @sila.UnobservableCommand(errors=[InstrumentBusy, NotAvailable])
    async def start_experiment(self, experiment: str) -> str:
        """
        Start an ImSwitch experiment, given as JSON in the form of ExperimentSchema (positions
        in µm, times in s, exposure in ms), and return at once. Follow it with
        ExperimentStatus; StopExperiment ends it.

        Args:
          Experiment: The experiment as JSON.

        Returns:
          Name: Name of the started experiment.
        """
        from imswitch.imcontrol.controller.controllers.experiment_controller import Experiment
        try:
            model = Experiment.model_validate_json(experiment)
        except ValueError as e:
            raise sila.errors.ValidationError(str(e), "Experiment") from None
        self._refuse_if_busy()
        await _call(self._experiment().startWellplateExperiment, model)
        return model.name

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_experiment_status(
        self,
    ) -> typing.Annotated[str, sila.constraints.Set(list(EXPERIMENT_STATES))]:
        """State of the experiment: idle, running, paused or stopping."""
        state = (await _call(self._experiment().getExperimentStatus)).get("status", "idle")
        return state if state in EXPERIMENT_STATES else "idle"

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_last_data_path(self) -> str:
        """Folder of the last experiment on the microscope; empty before the first."""
        return getattr(self._experiment(), "_last_experiment_dir", "") or ""

    @sila.UnobservableProperty()
    async def get_experiment_schema(self) -> str:
        """JSON schema of the experiment that StartExperiment takes."""
        from imswitch.imcontrol.controller.controllers.experiment_controller import Experiment
        return json.dumps(Experiment.model_json_schema())

    @sila.UnobservableCommand(errors=[NotAvailable])
    async def stop_experiment(self) -> None:
        """Stop the running experiment after the current image."""
        await _call(self._experiment().stopExperiment)

    @sila.UnobservableCommand(errors=[NotAvailable])
    async def pause_experiment(self) -> None:
        """Pause the running experiment."""
        await _call(self._experiment().pauseWorkflow)

    @sila.UnobservableCommand(errors=[NotAvailable])
    async def resume_experiment(self) -> None:
        """Resume a paused experiment."""
        await _call(self._experiment().resumeExperiment)


def _well_experiment(wells_xy, source, intensity, exposure_ms, autofocus, labware):
    """An Experiment that takes one image at the centre of each well."""
    from imswitch.imcontrol.controller.controllers.experiment_controller import Experiment
    return Experiment.model_validate({
        "name": "SiLA_ImageWells",
        "parameterValue": {
            "illumination": [source], "illuIntensities": [intensity],
            "exposureTimes": [exposure_ms], "timeLapsePeriod": 0, "numberOfImages": 1,
            "autoFocus": autofocus, "autoFocusMin": -50, "autoFocusMax": 50,
            "autoFocusStepSize": 5, "zStack": False, "zStackMin": 0, "zStackMax": 0,
        },
        "pointList": [
            {"name": wid, "x": x, "y": y, "wellId": wid, "labwareLoadName": labware}
            for wid, x, y in wells_xy
        ],
    })


# --- jobs (Aufgaben) ----------------------------------------------------------------------------

class JobController(_ImSwitchFeature):
    """
    Run saved ImSwitch jobs (Aufgaben): acquisitions composed and saved in the ImSwitch
    Aufgaben app (loops over wells and positions, channels, autofocus, analysis). A workcell
    step starts one by name and waits for it.
    """

    def _aufgaben(self):
        return self._controller("Aufgaben")

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_jobs(self) -> list[str]:
        """Names of the saved jobs and the bundled examples."""
        return [j["name"] for j in await _call(self._aufgaben().listAufgabenJobs)]

    @sila.UnobservableProperty(errors=[NotAvailable])
    async def get_job_status(self) -> typing.Annotated[str, sila.constraints.Set(list(JOB_STATES))]:
        """State of the current or last job: idle, running, paused, finished, aborted, failed."""
        state = (await _call(self._aufgaben().getAufgabenStatus)).get("state", "idle")
        return state if state in JOB_STATES else "idle"

    @sila.ObservableCommand(errors=[UnknownDevice, InstrumentBusy, NotConfirmed, JobFailed,
                                    NotAvailable])
    async def run_job(self, name: str, parameters: str, confirmed: bool, *,
                      status: sila.Status) -> str:
        """
        Run a saved job and wait until it ends. A job switches the light off when it ends; a
        macro (metadata.macro) leaves it as its steps left it.

        Args:
          Name: Job name, see Jobs.
          Parameters: Job parameters as a JSON object, e.g. {"exposure": 20}; empty for none.
          Confirmed: Accept the job's ask-first steps (homing, objective change). A job with
            such steps is refused without it.

        Returns:
          DataPath: Run folder on the microscope; empty when the job takes no images.
        """
        from imswitch.imcontrol.controller.controllers.AufgabenController import (
            StartAufgabenRequest)
        params = json.loads(parameters) if parameters.strip() else {}
        ab = self._aufgaben()
        job = await _call(ab.getAufgabenJob, name)
        request = StartAufgabenRequest(job=job, parameters=params, confirmed=confirmed)
        started = await _call(ab.startAufgabenJob, request)
        state = await self._wait_for_job(ab)
        if state.get("state") != "finished":
            raise JobFailed(f"Job '{name}' {state.get('state')}: {state.get('message', '')}")
        return started.get("folder") or ""

    @run_job.validator
    def _check_parameters(self, parameters: str) -> None:
        if not parameters.strip():
            return
        try:
            value = json.loads(parameters)
        except ValueError as e:
            raise sila.errors.ValidationError(f"Not JSON: {e}", "parameters") from None
        if not isinstance(value, dict):
            raise sila.errors.ValidationError("Expected a JSON object.", "parameters")

    async def _wait_for_job(self, ab) -> dict:
        while (state := await _call(ab.getAufgabenStatus)).get("state") in ("running", "paused"):
            if state.get("question"):
                await _call(ab.abortAufgabenJob, force=True)
                raise JobFailed(f"The job asked the operator: {state['question']}")
            await asyncio.sleep(0.5)
        return state

    @sila.UnobservableCommand(errors=[NotAvailable])
    async def abort_job(self) -> None:
        """Abort the running job, if any; the stage stops and the light goes off."""
        ab = self._aufgaben()
        if ab.isAufgabenRunning():
            await _call(ab.abortAufgabenJob, force=True)


# --- lock ---------------------------------------------------------------------------------------

class LockController(_cdk_lock.LockController):
    __doc__ = _cdk_lock.LockController.__doc__

    # The CDK's Lock.is_expired treats a timeout of 0 s as expired at once; the standard says
    # 0 s means no timeout.
    @property
    def lock(self):
        if self._lock is not None and not lock_active(self._lock):
            self._event.set()
            self._lock = None
        return self._lock

    @lock.setter
    def lock(self, value) -> None:
        self._event.set()
        self._lock = value

    async def on_after_start(self) -> None:
        await super().on_after_start()
        self._expiry = asyncio.get_running_loop().create_task(self._expire())

    async def _expire(self) -> None:
        """Tell IsLocked subscribers when a lock times out; the CDK noticed only on the next
        read."""
        while True:
            await asyncio.sleep(0.5)
            _ = self.lock


# --- plate hand-over (passive side) -------------------------------------------------------------

class LabwareTransferSiteController(_ImSwitchFeature):
    """
    Hand a plate over between a robot and the microscope stage (the passive side; the robot
    implements Labware Transfer Manipulator Controller).

    Delivering a plate: PrepareForInput (the stage moves to its loading position and the light
    goes off), the robot puts the plate, then LabwareDelivered. Taking it: PrepareForOutput, the
    robot takes the plate, then LabwareRemoved. The microscope has one handover position (the
    stage insert) and one internal position.
    """

    def __init__(self, master, position_name: str = "Stage"):
        # The identifier of the SiLA standard feature, so a scheduler finds it.
        super().__init__(master, originator="org.silastandard",
                         category="instruments.labware.transfer")
        self._position_name = position_name
        self._expected = ""   # "input" or "output" after the matching Prepare command
        self._pending = ("", "")
        self.labware = ("", "")  # (type, unique id) of the plate on the stage; ("", "") = none

    def _check_position(self, handover_position: HandoverPosition) -> None:
        if handover_position.position != self._position_name:
            raise sila.errors.ValidationError(
                f"Unknown handover position '{handover_position.position}'; "
                f"this microscope has '{self._position_name}'.", "handover_position")
        if handover_position.sub_position.position_index != 1:
            raise sila.errors.ValidationError("The handover position has one sub-position, 1.",
                                              "handover_position")

    def _check_positions(self, handover_position: HandoverPosition,
                         internal_position: PositionIndex) -> None:
        self._check_position(handover_position)
        if internal_position.position_index != 1:
            raise sila.errors.ValidationError("The microscope has one internal position, 1.",
                                              "internal_position")

    async def _to_loading_position(self, status) -> None:
        self._refuse_if_busy()
        await asyncio.to_thread(self._all_light_off)
        status.update(progress=0.1)
        await _call(self._controller("Positioner").moveToSampleLoadingPosition, is_blocking=True)

    @sila.UnobservableProperty()
    async def get_available_handover_positions(self) -> list[HandoverPosition]:
        """All handover positions of the device including the number of sub-positions."""
        return [HandoverPosition(position=self._position_name,
                                 sub_position=PositionIndex(position_index=1))]

    @sila.UnobservableProperty()
    async def get_number_of_internal_positions(
        self,
    ) -> typing.Annotated[int, sila.constraints.MinimalInclusive(value=1)]:
        """Get the number of addressable internal positions of the device."""
        return 1

    @sila.ObservableCommand(errors=[InvalidCommandSequence, InstrumentBusy, NotAvailable])
    async def prepare_for_input(
        self,
        handover_position: HandoverPosition,
        internal_position: PositionIndex,
        labware_type: str,
        labware_unique_id: str,
        *,
        status: sila.Status,
    ) -> None:
        """
        Put the device into a state in which it is ready to accept new labware at the specified
        handover position: the light goes off and the stage moves to its loading position.

        Args:
          HandoverPosition: Indicates the position where the labware will be handed over.
          InternalPosition: Indicates the position which the labware will be stored at within
            the device; always 1.
          LabwareType: Specifies the type of labware that will be handed over.
          LabwareUniqueID: Represents the unique identification of a labware in the
            controlling system.
        """
        if self.labware[0] or self.labware[1]:
            raise InvalidCommandSequence(f"A plate ({self.labware[1] or self.labware[0]}) is "
                                         "still on the stage.")
        await self._to_loading_position(status)
        self._expected, self._pending = "input", (labware_type, labware_unique_id)

    @sila.ObservableCommand(errors=[InvalidCommandSequence, InstrumentBusy, NotAvailable])
    async def prepare_for_output(
        self,
        handover_position: HandoverPosition,
        internal_position: PositionIndex,
        *,
        status: sila.Status,
    ) -> None:
        """
        Put the device into a state in which it is ready to release the labware at the specified
        handover position: the light goes off and the stage moves to its loading position.

        Args:
          HandoverPosition: Indicates the position where the labware will be handed over.
          InternalPosition: Indicates the position which the labware will be retrieved from
            within the device; always 1.
        """
        await self._to_loading_position(status)
        self._expected = "output"

    prepare_for_input.validator(_check_positions)
    prepare_for_output.validator(_check_positions)

    @sila.UnobservableCommand(errors=[InvalidCommandSequence])
    async def labware_delivered(self, handover_position: HandoverPosition) -> None:
        """
        Notify the device that the labware item has been placed at the handover position.

        Args:
          HandoverPosition: Indicates the position where the labware has been handed over.
        """
        self._check_position(handover_position)
        if self._expected != "input":
            raise InvalidCommandSequence("Prepare For Input was not called.")
        self.labware, self._expected = self._pending, ""

    @sila.UnobservableCommand(errors=[InvalidCommandSequence])
    async def labware_removed(self, handover_position: HandoverPosition) -> None:
        """
        Notify the device that the labware item has been removed from the handover position.

        Args:
          HandoverPosition: Indicates the position where the labware has been handed over.
        """
        self._check_position(handover_position)
        if self._expected != "output":
            raise InvalidCommandSequence("Prepare For Output was not called.")
        self.labware, self._expected = ("", ""), ""


def create_features(master, handover_position_name: str = "Stage", plate_origins=None,
                    save_setup=None) -> list:
    """The features SiLa2Controller registers, the SiLA standard Lock Controller last."""
    global _lock_feature
    _lock_feature = LockController()
    return [
        StageController(master),
        ImagingController(master),
        ExperimentController(master, origins=plate_origins, save=save_setup),
        JobController(master),
        LabwareTransferSiteController(master, handover_position_name),
        _lock_feature,
    ]
