"""Pydantic models for the labware layer.

Two layers:
  - ``OpentronsLabwareV2``: 1:1 with Opentrons schema v2 JSON. mm units. Used
    only inside the loader.
  - ``LabwareDefinition``: the in-memory model the rest of ImSwitch consumes.
    micrometres (µm) everywhere.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Opentrons-compat layer (mm).  Used only inside the loader.
# ---------------------------------------------------------------------------


class _Vector(BaseModel):
    x: float
    y: float
    z: float


class _BrandData(BaseModel):
    brand: str
    brandId: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)


class _LabwareMetadata(BaseModel):
    displayName: str
    displayCategory: str
    displayVolumeUnits: Literal["µL", "mL", "L"]
    tags: List[str] = Field(default_factory=list)


class _LabwareParameters(BaseModel):
    format: Literal["96Standard", "384Standard", "trough", "irregular", "trash"]
    isTiprack: bool
    loadName: str
    isMagneticModuleCompatible: bool = False
    quirks: List[str] = Field(default_factory=list)
    isDeckSlotCompatible: bool = True

    model_config = ConfigDict(extra="allow")


class _LabwareDimensions(BaseModel):
    xDimension: float
    yDimension: float
    zDimension: float


class OpentronsLabwareV2(BaseModel):
    """1:1 mirror of Opentrons labware schema v2.  mm.  Loader-only."""

    schemaVersion: Literal[2]
    version: int
    namespace: str
    metadata: _LabwareMetadata
    brand: _BrandData
    parameters: _LabwareParameters
    cornerOffsetFromSlot: _Vector
    ordering: List[List[str]]
    dimensions: _LabwareDimensions
    # Wells are validated structurally in the loader so we can produce
    # well-id-aware error messages.
    wells: Dict[str, Dict[str, Any]]
    groups: List[Dict[str, Any]] = Field(default_factory=list)

    # Tolerate unknown future fields (Opentrons keeps extending the schema).
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# ImSwitch in-memory layer (µm).  This is what the rest of ImSwitch sees.
# ---------------------------------------------------------------------------


class WellGeometry(BaseModel):
    """Geometry of a single well, in µm."""

    shape: Literal["circle", "rectangle"]
    radius: Optional[float] = None         # circle, µm
    width: Optional[float] = None          # rectangle, µm
    height: Optional[float] = None         # rectangle, µm
    depth: float                           # well depth top -> bottom, µm
    totalLiquidVolume_uL: float            # microlitres (volume is not a length)


class WellPosition(BaseModel):
    """Position of a single well in plate coordinates, µm."""

    id: str
    row: str
    column: int
    x: float    # µm, well center, plate frame
    y: float    # µm
    z: float    # µm, center-bottom of well, plate frame
    geometry: WellGeometry


class LabwareDimensions(BaseModel):
    """Outer dimensions of the labware, µm."""

    x: float
    y: float
    z: float


class LabwareDefinition(BaseModel):
    """In-memory labware definition. µm everywhere."""

    schema_version: int = 1
    load_name: str
    namespace: str
    version: int
    display_name: str
    display_category: str
    format: str
    brand: str
    brand_ids: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    dimensions: LabwareDimensions
    rows: List[str]
    columns: List[int]
    ordering: List[List[str]]                 # columns of well IDs (Opentrons convention)
    wells: Dict[str, WellPosition]
    well_names_flat: List[str]                # row-major flat list
    groups: List[Dict[str, Any]] = Field(default_factory=list)
    corner_offset_from_slot_um: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    source_path: Optional[str] = None

    def well_ids(self) -> List[str]:
        return list(self.wells.keys())

    def get_well(self, well_id: str) -> WellPosition:
        try:
            return self.wells[well_id]
        except KeyError as exc:
            raise KeyError(
                f"Well {well_id!r} not in labware {self.load_name!r}"
            ) from exc


__all__ = [
    "OpentronsLabwareV2",
    "WellGeometry",
    "WellPosition",
    "LabwareDimensions",
    "LabwareDefinition",
]
