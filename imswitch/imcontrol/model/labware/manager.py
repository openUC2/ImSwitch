"""LabwareManager: discovers, validates and caches labware definitions."""

from __future__ import annotations


import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from .loader import LabwareValidationError, load_labware_from_file
from .models import LabwareDefinition

logger = logging.getLogger(__name__)


class LabwareManager:
    """Loads and caches all labware definitions in a directory tree.

    A bad labware file is skipped with a WARN — never crashes startup.
    """

    def __init__(self, definitions_root: Optional[Union[str, Path]] = None) -> None:
        if definitions_root is None:
            definitions_root = Path(__file__).parent / "definitions"
        self._root = Path(definitions_root)
        self._defs: Dict[str, LabwareDefinition] = {}
        self.reload()

    @property
    def definitions_root(self) -> Path:
        return self._root

    def reload(self) -> None:
        """Rescan the filesystem for labware JSON files."""
        self._defs.clear()
        if not self._root.exists():
            logger.warning("Labware definitions root does not exist: %s", self._root)
            return
        for path in sorted(self._root.rglob("*.json")):
            # Skip optional registry/index files.
            if path.name.startswith("_"):
                continue
            try:
                lab = load_labware_from_file(path)
            except LabwareValidationError as exc:
                logger.warning("Skipping invalid labware %s: %s", path, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Unexpected error loading %s: %s", path, exc)
                continue
            if lab.load_name in self._defs:
                logger.warning(
                    "Duplicate labware loadName %r at %s — keeping first (%s)",
                    lab.load_name,
                    path,
                    self._defs[lab.load_name].source_path,
                )
                continue
            self._defs[lab.load_name] = lab
        logger.info("LabwareManager loaded %d definitions from %s",
                    len(self._defs), self._root)

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------
    def list_load_names(self) -> List[str]:
        return sorted(self._defs.keys())

    def list_summaries(self) -> List[dict]:
        out: List[dict] = []
        for name in self.list_load_names():
            lab = self._defs[name]
            out.append({
                "load_name": lab.load_name,
                "display_name": lab.display_name,
                "display_category": lab.display_category,
                "format": lab.format,
                "rows": len(lab.rows),
                "cols": len(lab.columns),
                "well_count": len(lab.wells),
                "dimensions": {
                    "x_um": lab.dimensions.x,
                    "y_um": lab.dimensions.y,
                    "z_um": lab.dimensions.z,
                },
                "brand": lab.brand,
                "tags": list(lab.tags),
                "namespace": lab.namespace,
                "version": lab.version,
            })
        return out

    def get(self, load_name: str) -> LabwareDefinition:
        try:
            return self._defs[load_name]
        except KeyError as exc:
            raise KeyError(
                f"Labware {load_name!r} not loaded. Known: {self.list_load_names()}"
            ) from exc

    def register(self, lab: LabwareDefinition, *, overwrite: bool = False) -> None:
        """Register an in-memory labware (e.g. ephemeral, generator-built)."""
        if not overwrite and lab.load_name in self._defs:
            raise ValueError(
                f"Labware {lab.load_name!r} already registered"
            )
        self._defs[lab.load_name] = lab

    def get_with_offset(
        self, load_name: str, offset_x_um: float, offset_y_um: float
    ) -> LabwareDefinition:
        """Return a deep-copied labware with x/y offsets added to every well."""
        base = self.get(load_name)
        if offset_x_um == 0 and offset_y_um == 0:
            return base.model_copy(deep=True)
        clone = base.model_copy(deep=True)
        for wid, well in clone.wells.items():
            clone.wells[wid] = well.model_copy(update={
                "x": well.x + offset_x_um,
                "y": well.y + offset_y_um,
            })
        return clone


__all__ = ["LabwareManager"]
