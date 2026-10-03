"""Labware and well-selection endpoints.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
from fastapi import HTTPException
from typing import List, Dict


from imswitch.imcommon.model import APIExport


class LabwareMixin:
    """Labware and well-selection endpoints."""

    @staticmethod
    def _labware_to_layout_dict(lab, offset_x: float = 0.0, offset_y: float = 0.0) -> dict:
        """Convert a ``LabwareDefinition`` to the canvas-style wellplate
        layout dict used by the overview-camera registration code. µm
        everywhere."""
        spacing_x = 0.0
        spacing_y = 0.0
        if len(lab.columns) >= 2:
            r = lab.rows[0]
            w_a = lab.wells.get(f"{r}{lab.columns[0]}")
            w_b = lab.wells.get(f"{r}{lab.columns[1]}")
            if w_a and w_b:
                spacing_x = abs(w_b.x - w_a.x)
        if len(lab.rows) >= 2:
            c = lab.columns[0]
            w_a = lab.wells.get(f"{lab.rows[0]}{c}")
            w_b = lab.wells.get(f"{lab.rows[1]}{c}")
            if w_a and w_b:
                spacing_y = abs(w_b.y - w_a.y)

        wells_out = []
        for wid in lab.well_names_flat:
            w = lab.wells[wid]
            entry = {
                "id": wid,
                "name": wid,
                "x": w.x + offset_x,
                "y": w.y + offset_y,
                "shape": w.geometry.shape,
                "row": lab.rows.index(w.row),
                "col": lab.columns.index(w.column),
            }
            if w.geometry.shape == "circle":
                entry["radius"] = w.geometry.radius
            else:
                entry["width"] = w.geometry.width
                entry["height"] = w.geometry.height
            wells_out.append(entry)
        return {
            "name": lab.display_name,
            "description": ", ".join(lab.tags) if lab.tags else "",
            "rows": len(lab.rows),
            "cols": len(lab.columns),
            "well_spacing_x": spacing_x,
            "well_spacing_y": spacing_y,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "wells": wells_out,
            "unit": "um",
            "width": lab.dimensions.x,
            "height": lab.dimensions.y,
            "labwareLoadName": lab.load_name,
        }

    # ------------------------------------------------------------------
    # New Opentrons-style endpoints
    # ------------------------------------------------------------------
    @APIExport(requestType="GET")
    def getLabwareList(self) -> List[dict]:
        """List summaries of all loaded labware definitions (µm dimensions)."""
        if not self.labware_manager:
            return []
        return self.labware_manager.list_summaries()

    @APIExport(requestType="GET")
    def getLabwareDefinition(
        self,
        load_name: str,
        offset_x_um: float = 0.0,
        offset_y_um: float = 0.0,
    ) -> dict:
        """Return a full ``LabwareDefinition`` (µm). Optional offsets shift
        every well's x/y to map plate -> stage coordinates."""
        if not self.labware_manager:
            raise HTTPException(status_code=503, detail="LabwareManager unavailable")
        try:
            lab = self.labware_manager.get_with_offset(load_name, offset_x_um, offset_y_um)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return lab.model_dump()

    @APIExport(requestType="POST")
    def selectWellsByPattern(self, request: dict) -> dict:
        """Resolve a ``WellSelectionPattern`` against a labware. Pure function;
        does not mutate experiment state.

        Request shape:
            {
              "load_name": "corning_96_wellplate_360ul_flat",
              "pattern": {"ranges": ["A1:H12"], "rows": ["A"], ...},
              "offset_x_um": 0.0,
              "offset_y_um": 0.0
            }
        """
        if not self.labware_manager:
            raise HTTPException(status_code=503, detail="LabwareManager unavailable")
        from imswitch.imcontrol.model.labware import (
            WellSelectionPattern, resolve_pattern,
        )
        load_name = request.get("load_name")
        pattern_raw = request.get("pattern") or {}
        offset_x = float(request.get("offset_x_um") or 0.0)
        offset_y = float(request.get("offset_y_um") or 0.0)
        try:
            lab = self.labware_manager.get(load_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            pattern = WellSelectionPattern.model_validate(pattern_raw)
            wells = resolve_pattern(lab, pattern)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        resolved = []
        for w in wells:
            entry = {
                "well_id": w.id,
                "row": w.row,
                "column": w.column,
                "x_um": w.x + offset_x,
                "y_um": w.y + offset_y,
                "z_um": w.z,
                "shape": w.geometry.shape,
            }
            if w.geometry.shape == "circle":
                entry["radius_um"] = w.geometry.radius
            else:
                entry["width_um"] = w.geometry.width
                entry["height_um"] = w.geometry.height
            resolved.append(entry)
        return {"load_name": load_name, "count": len(resolved), "wells": resolved}

    @APIExport(requestType="POST")
    def applyWellSelectionToExperiment(self, request: dict) -> dict:
        """Resolve a pattern and return ``Point`` dicts ready for the
        experiment's pointList.

        Request shape:
            {
              "load_name": "...",
              "pattern": {...},
              "offset_x_um": 0,
              "offset_y_um": 0,
              "condition_labels": {"A1": "Donor1+DMSO", ...},
              "point_name_template": "{well_id}"
            }
        """
        if not self.labware_manager:
            raise HTTPException(status_code=503, detail="LabwareManager unavailable")
        from imswitch.imcontrol.model.labware import (
            WellSelectionPattern, resolve_pattern,
        )
        load_name = request.get("load_name")
        pattern_raw = request.get("pattern") or {}
        offset_x = float(request.get("offset_x_um") or 0.0)
        offset_y = float(request.get("offset_y_um") or 0.0)
        condition_labels: Dict[str, str] = dict(request.get("condition_labels") or {})
        template: str = request.get("point_name_template") or "{well_id}"
        try:
            lab = self.labware_manager.get(load_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            pattern = WellSelectionPattern.model_validate(pattern_raw)
            wells = resolve_pattern(lab, pattern)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        points: List[dict] = []
        for w in wells:
            label = condition_labels.get(w.id)
            try:
                name = template.format(
                    well_id=w.id, row=w.row, column=w.column, label=label or ""
                )
            except (KeyError, IndexError):
                name = w.id
            points.append({
                "name": name,
                "x": w.x + offset_x,
                "y": w.y + offset_y,
                "z": w.z if w.z != 0 else None,
                "iX": 0,
                "iY": 0,
                "neighborPointList": [],
                "wellId": w.id,
                "wellRow": w.row,
                "wellColumn": w.column,
                "labwareLoadName": load_name,
                "conditionLabel": label,
                "areaType": "well",
            })
        return {"load_name": load_name, "count": len(points), "points": points}
