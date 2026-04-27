"""
Pydantic data models for the ExperimentController.

These models define the JSON contract used by the React frontend (see
``frontend/src/backendapi/apiExperimentController*.js``) and are exported
back from :mod:`ExperimentController` for backwards compatibility with
``from imswitch.imcontrol.controller.controllers.ExperimentController
import Experiment`` style imports.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# Pydantic v1/v2 compatibility for field-level validators.
try:  # pragma: no cover - exercised by version of pydantic installed
    from pydantic import field_validator  # pydantic >= 2
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import validator as _v1_validator  # pydantic 1.x
    _PYDANTIC_V2 = False

    def field_validator(*fields, mode: str = "after"):  # type: ignore[no-redef]
        """Minimal shim mapping v2 ``field_validator`` to v1 ``validator``."""
        return _v1_validator(*fields, pre=(mode == "before"), allow_reuse=True)


# ---------------------------------------------------------------------------
# Literal aliases for fields that previously accepted any string.  Pydantic
# rejects unknown values with a 422 at the API boundary, giving the React
# client a clear error instead of silent misbehavior downstream.
# ---------------------------------------------------------------------------
KeepIlluminationMode = Literal["auto", "on", "off"]
AutoFocusMode = Literal["software", "hardware"]
AutoFocusSoftwareMethod = Literal["scan", "hillClimbing"]
TriggerMode = Literal["hardware", "software"]
FocusFitMethod = Literal["spline", "rbf", "constant"]
FocusAlgorithm = Literal["LAPE", "GLVA", "JPEG"]
ScanPattern = Literal["raster", "snake"]


# ---------------------------------------------------------------------------
# Focus map request models
# ---------------------------------------------------------------------------
class FocusMapFromPointsRequest(BaseModel):
    """Request body for ``computeFocusMapFromPoints`` endpoint."""
    points: List[Dict[str, float]]
    group_id: str = "manual"
    group_name: str = "Manual Points"
    method: FocusFitMethod = "rbf"
    smoothing_factor: float = 0.1
    z_offset: float = 0.0
    clamp_enabled: bool = False
    z_min: float = 0.0
    z_max: float = 0.0


# ---------------------------------------------------------------------------
# Point / scan-area models
# ---------------------------------------------------------------------------
class NeighborPoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = None
    iX: int
    iY: int


class Point(BaseModel):
    id: Optional[str] = None  # Allow string IDs from frontend
    name: str
    x: float
    y: float
    z: Optional[float] = None  # Per-point Z origin for Z-stacking/autofocus
    iX: int = 0
    iY: int = 0
    neighborPointList: List[NeighborPoint] = Field(default_factory=list)
    wellId: Optional[str] = None
    areaType: Optional[str] = None  # well, free_scan, etc.


class ScanPosition(BaseModel):
    """Single position in a scan area."""
    index: int
    x: float
    y: float
    z: Optional[float] = None  # Per-position Z origin
    iX: int
    iY: int


class ScanBounds(BaseModel):
    """Bounding box for a scan area."""
    minX: float
    maxX: float
    minY: float
    maxY: float
    width: float
    height: float


class CenterPosition(BaseModel):
    """Center position of a scan area."""
    x: float
    y: float
    z: Optional[float] = None


class ScanArea(BaseModel):
    """Pre-calculated scan area with ordered positions."""
    areaId: str
    areaName: str
    areaType: str = "free_scan"
    wellId: Optional[str] = None
    centerPosition: CenterPosition
    bounds: ScanBounds
    scanPattern: ScanPattern = "raster"
    positions: List[ScanPosition]


class ScanMetadata(BaseModel):
    """Metadata for the entire scan."""
    totalPositions: int
    fovX: float
    fovY: float
    overlapWidth: float = 0.0
    overlapHeight: float = 0.0
    scanPattern: ScanPattern = "raster"


# ---------------------------------------------------------------------------
# Parameter / experiment models
# ---------------------------------------------------------------------------
class ParameterValue(BaseModel):
    illumination: Union[List[str], str] = None
    illuIntensities: Union[List[Optional[int]], Optional[int]] = None
    brightfield: bool = False
    darkfield: bool = False
    differentialPhaseContrast: bool = False
    timeLapsePeriod: float
    numberOfImages: int
    autoFocus: bool
    autoFocusMin: float
    autoFocusMax: float
    autoFocusStepSize: float
    autoFocusIlluminationChannel: str = ""
    autoFocusMode: AutoFocusMode = "software"
    autoFocusSoftwareMethod: AutoFocusSoftwareMethod = "scan"
    autoFocusHillClimbingInitialStep: float = 20.0
    autoFocusHillClimbingMinStep: float = 1.0
    autoFocusHillClimbingStepReduction: float = 0.5
    autoFocusHillClimbingMaxIterations: int = 50
    autofocus_target_focus_setpoint: Optional[float] = None
    autofocus_max_attempts: int = 2
    zStack: bool
    zStackMin: float
    zStackMax: float
    zStackStepSize: Union[List[float], float] = 1.0
    exposureTimes: Union[List[float], float] = None
    gains: Union[List[float], float] = None
    speed: float = 20000.0
    performanceMode: bool = False
    performanceTriggerMode: TriggerMode = Field(
        "hardware",
        description="Trigger mode: 'hardware' (external TTL) or 'software' (callback-based)",
    )
    performanceTPreMs: float = Field(90.0, description="Pre-exposure settle time in milliseconds")
    performanceTPostMs: float = Field(50.0, description="Post-exposure/acquisition time in milliseconds")
    ome_write_tiff: bool = Field(False, description="Whether to write OME-TIFF files")
    ome_write_zarr: bool = Field(True, description="Whether to write OME-Zarr files")
    ome_write_stitched_tiff: bool = Field(False, description="Whether to write stitched OME-TIFF files")
    ome_write_individual_tiffs: bool = Field(False, description="Whether to write individual TIFF files per frame")
    ome_write_single_tiff: bool = Field(
        False,
        description="Whether to write a single OME-TIFF (auto-enabled for single-tile scans)",
    )
    keepIlluminationOn: KeepIlluminationMode = Field(
        "auto",
        description="Illumination mode: 'auto' (single channel stays on), 'on' (always on), 'off' (per-frame toggle)",
    )

    # ------------------------------------------------------------------
    # Validators – run once at the API boundary so the controller can
    # treat list-typed fields as guaranteed lists.
    # ------------------------------------------------------------------
    @field_validator(
        "illumination", "illuIntensities", "gains", "exposureTimes",
        mode="before",
    )
    @classmethod
    def _coerce_to_list(cls, v):
        if v is None:
            return v
        return v if isinstance(v, list) else [v]

    # ------------------------------------------------------------------
    # Convenience properties – move trivial computations off the
    # controller so callers read intent instead of re-deriving it.
    # ------------------------------------------------------------------
    @property
    def performance_t_pre_s(self) -> float:
        """Pre-exposure settle time in seconds."""
        return self.performanceTPreMs / 1000.0

    @property
    def performance_t_post_s(self) -> float:
        """Post-exposure settle time in seconds."""
        return self.performanceTPostMs / 1000.0

    @property
    def passthrough_illumination(self) -> bool:
        """True when no illumination intensities are configured.

        In passthrough mode, the acquisition loop runs without changing the
        device's current illumination/exposure/gain settings.
        """
        return not any(self.illuIntensities or [])

    @property
    def n_active_channels(self) -> int:
        """Number of illumination channels with intensity > 0."""
        return sum(1 for v in (self.illuIntensities or []) if v and v > 0)

    def resolve_keep_illumination_on(self) -> bool:
        """Resolve the auto/on/off setting to a concrete bool."""
        if self.keepIlluminationOn == "on":
            return True
        if self.keepIlluminationOn == "off":
            return False
        # "auto" → keep on iff exactly one active channel
        return self.n_active_channels == 1


class FocusMapConfig(BaseModel):
    """Configuration for optional focus mapping (Z surface estimation over XY)."""
    enabled: bool = Field(False, description="Enable focus mapping before acquisition")

    # Grid generation
    rows: int = Field(3, description="Number of grid rows for focus measurement")
    cols: int = Field(3, description="Number of grid columns for focus measurement")
    add_margin: bool = Field(False, description="Shrink grid inward to avoid edge effects")

    # Fit strategy
    fit_by_region: bool = Field(True, description="Fit per well / scan region (True) or global (False)")
    use_manual_map: bool = Field(False, description="Reuse a pre-existing manual/global map for all groups via interpolation instead of measuring per group")
    method: FocusFitMethod = Field("spline", description="Fit method: spline, rbf, or constant")
    smoothing_factor: float = Field(0.1, description="Smoothing factor for surface fit")

    # Runtime behavior
    apply_during_scan: bool = Field(True, description="Move Z per XY using focus map during acquisition")
    z_offset: float = Field(0.0, description="Global Z offset applied to interpolated values")
    clamp_enabled: bool = Field(False, description="Clamp interpolated Z to min/max range")
    z_min: float = Field(0.0, description="Minimum allowed Z value when clamping")
    z_max: float = Field(0.0, description="Maximum allowed Z value when clamping")

    # Autofocus integration
    autofocus_profile: Optional[str] = Field(None, description="Reference to AF controller preset")
    settle_ms: int = Field(0, description="Extra settle time in ms after Z move")
    store_debug_artifacts: bool = Field(True, description="Store focus points + fit stats as JSON")
    channel_offsets: Optional[Dict[str, float]] = Field(default=None, description="Per-illumination-channel Z offset (µm)")

    # Autofocus parameters – passed through to doAutofocusBackground
    af_range: float = Field(100.0, description="Autofocus Z range (±µm from current Z)")
    af_resolution: float = Field(10.0, description="Autofocus step size (µm)")
    af_cropsize: int = Field(2048, description="Crop size for focus quality algorithm")
    af_algorithm: FocusAlgorithm = Field("LAPE", description="Focus quality algorithm: LAPE, GLVA, JPEG")
    af_settle_time: float = Field(0.1, description="Settle time (s) after each Z step")
    af_static_offset: float = Field(0.0, description="Static Z offset applied after autofocus (µm)")
    af_two_stage: bool = Field(False, description="Use two-stage autofocus (coarse + fine)")
    af_n_gauss: int = Field(0, description="Gaussian kernel size for focus algorithm")
    af_illumination_channel: str = Field("", description="Illumination channel for autofocus")
    af_mode: AutoFocusMode = Field("software", description="Autofocus mode: software (Z-sweep) or hardware (FocusLock)")
    af_software_method: AutoFocusSoftwareMethod = Field("scan", description="Software AF method: scan (Z-sweep) or hillClimbing")
    af_hc_initial_step: float = Field(20.0, description="Hill climbing initial step size (µm)")
    af_hc_min_step: float = Field(1.0, description="Hill climbing minimum step size (µm)")
    af_hc_step_reduction: float = Field(0.5, description="Hill climbing step reduction factor")
    af_hc_max_iterations: int = Field(50, description="Hill climbing max iterations")
    af_max_attempts: int = Field(2, description="Max retry attempts for hardware autofocus")
    af_target_setpoint: Optional[float] = Field(None, description="Target focus setpoint for hardware AF")

    # Scan areas – passed from the frontend so that computeFocusMap knows the
    # correct XY bounds even when no experiment has been started yet.
    scan_areas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of scan area dicts with areaId, areaName, bounds (minX/maxX/minY/maxY)",
    )


class Experiment(BaseModel):
    name: str
    parameterValue: ParameterValue
    pointList: List[Point] = Field(default_factory=list)

    # Pre-calculated scan data from frontend
    scanAreas: Optional[List[ScanArea]] = None
    scanMetadata: Optional[ScanMetadata] = None

    # Focus mapping configuration (disabled by default)
    focusMap: Optional[FocusMapConfig] = Field(default=None, description="Optional focus mapping configuration")

    timepoints: int = Field(1, description="Number of timepoints for time-lapse")

    def to_configuration(self) -> dict:
        """Convert this Experiment into a dict structure for downstream consumers."""
        return {
            "experiment": {
                "MicroscopeState": {
                    "timepoints": self.timepoints,
                },
            },
        }


# ---------------------------------------------------------------------------
# MDA-related models for useq-schema integration
# ---------------------------------------------------------------------------
class MDAChannelConfig(BaseModel):
    """Configuration for an MDA channel."""
    name: str = Field(..., description="Channel name/identifier")
    exposure: Optional[float] = Field(100.0, description="Exposure time in milliseconds")
    power: Optional[float] = Field(100.0, description="Laser/illumination power")


class MDASequenceRequest(BaseModel):
    """Request to start an MDA experiment using useq-schema."""
    channels: List[MDAChannelConfig] = Field(..., description="List of channel configurations")
    z_range: Optional[float] = Field(None, description="Total Z range to scan (µm)")
    z_step: Optional[float] = Field(None, description="Z step size (µm)")
    time_points: int = Field(1, description="Number of time points")
    time_interval: float = Field(1.0, description="Interval between time points (seconds)")
    save_directory: Optional[str] = Field(None, description="Directory to save data")
    experiment_name: str = Field("MDA_Experiment", description="Name of the experiment")


class MDASequenceInfo(BaseModel):
    """Information about an MDA sequence."""
    total_events: int
    channels: List[str]
    z_positions: List[float]
    time_points: List[int]
    axis_order: tuple
    estimated_duration_minutes: float


class ExperimentWorkflowParams(BaseModel):
    """Parameters describing the controller's hardware capabilities."""

    # Illumination parameters
    illuSources: List[str] = Field(default_factory=list, description="List of illumination sources")
    illuSourceMinIntensities: List[float] = Field(default_factory=list, description="Minimum intensities for each source")
    illuSourceMaxIntensities: List[float] = Field(default_factory=list, description="Maximum intensities for each source")
    illuIntensities: List[float] = Field(default_factory=list, description="Intensities for each source")

    # Camera parameters
    exposureTimes: List[float] = Field(default_factory=list, description="Exposure times for each source")
    gains: List[float] = Field(default_factory=list, description="gains settings for each source")

    # Feature toggles
    isDPCpossible: bool = Field(False, description="Whether DPC is possible")
    isDarkfieldpossible: bool = Field(False, description="Whether darkfield is possible")

    # timelapse parameters
    timeLapsePeriodMin: float = Field(0, description="Minimum time for a timelapse series")
    timeLapsePeriodMax: float = Field(100000000, description="Maximum time for a timelapse series in seconds")
    numberOfImagesMin: int = Field(0, description="Minimum number of images for a timelapse series")
    numberOfImagesMax: int = Field(0, description="Maximum number of images for a timelapse series")
    autofocusMinFocusPosition: float = Field(-10000, description="Minimum autofocus position")
    autofocusMaxFocusPosition: float = Field(10000, description="Maximum autofocus position")
    autofocusStepSizeMin: float = Field(1, description="Minimum autofocus step size")
    autofocusStepSizeMax: float = Field(1000, description="Maximum autofocus step size")
    zStackMinFocusPosition: float = Field(0, description="Minimum Z-stack position")
    zStackMaxFocusPosition: float = Field(10000, description="Maximum Z-stack position")
    zStackStepSizeMin: float = Field(1, description="Minimum Z-stack step size")
    zStackStepSizeMax: float = Field(1000, description="Maximum Z-stack step size")
    performanceMode: bool = Field(
        False,
        description=(
            "Whether to use performance mode for the experiment - executing the scan "
            "on the C++ hardware directly rather than on the Python side."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint response models
# ---------------------------------------------------------------------------
class StartExperimentResponse(BaseModel):
    """Typed response for ``startWellplateExperiment``.

    Kept compatible with the historical raw-dict shape of
    ``{"status": "running", "mode": ...}`` while adding optional fields
    that the React client can ignore.
    """
    status: Literal["running", "queued", "rejected"] = "running"
    mode: Literal["normal", "performance"] = "normal"
    started_at: Optional[datetime] = None


__all__ = [
    # Literal aliases
    "KeepIlluminationMode",
    "AutoFocusMode",
    "AutoFocusSoftwareMethod",
    "TriggerMode",
    "FocusFitMethod",
    "FocusAlgorithm",
    "ScanPattern",
    # Models
    "FocusMapFromPointsRequest",
    "NeighborPoint",
    "Point",
    "ScanPosition",
    "ScanBounds",
    "CenterPosition",
    "ScanArea",
    "ScanMetadata",
    "ParameterValue",
    "FocusMapConfig",
    "Experiment",
    "MDAChannelConfig",
    "MDASequenceRequest",
    "MDASequenceInfo",
    "ExperimentWorkflowParams",
    "StartExperimentResponse",
]
