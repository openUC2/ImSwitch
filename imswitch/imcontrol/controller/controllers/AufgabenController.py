"""Aufgaben: composable acquisition jobs over HTTP.

Jobs are documents (``imswitch/imcontrol/model/aufgaben``) that the engine runs
on this microscope through ``ImSwitchMicroscope``. Saved jobs live in
``<config>/aufgaben/*.json``; each run writes OME-Zarr (and optional OME-TIFF),
``job.json`` and ``run.jsonl`` to ``<data>/Aufgaben/<timestamp>_<job>/``.
Events (state, node, log, question) go out on ``sigAufgabenEvent``.
"""
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcontrol.model import aufgaben
from imswitch.imcontrol.model.aufgaben.engine import Run
from imswitch.imcontrol.model.aufgaben.storage import OmeStore
from ..basecontrollers import ImConWidgetController
from ..busy import busy_reasons
from .aufgaben import ImSwitchMicroscope
from .aufgaben.host import register_labware

ACTIVE = ("running", "paused")
DEVICE_KINDS = ("illumination", "detector", "digital_output", "digital_input",
                "temperature", "pump", "valve")
NODE_EVENT_PERIOD_S = 0.1


class StartAufgabenRequest(BaseModel):
    job: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = Field(False, description="The operator accepted the ask-first steps")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "job"


class AufgabenController(ImConWidgetController):
    sigAufgabenEvent = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._host = ImSwitchMicroscope(self._master, self._logger)
        self._run: Optional[Run] = None
        self._lock = threading.Lock()
        self._last_node_event = 0.0
        self._jobs_dir = Path(dirtools.UserFileDirs.Root) / "aufgaben"
        self._plates = []   # ImSwitch labware usable as useq plate names in jobs
        if self._labware() is not None:
            try:
                self._plates = register_labware(self._labware())
            except Exception as e:
                self._logger.warning(f"Labware not registered as useq plates: {e}")

    def _labware(self):
        return getattr(self._master.getController("Experiment"), "labware_manager", None)

    # -- helpers --------------------------------------------------------------
    def _job(self, data: Dict[str, Any]) -> aufgaben.Job:
        try:
            return aufgaben.Job.model_validate(data)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=json.loads(e.json())) from e

    def _runs_dir(self) -> Path:
        return Path(dirtools.UserFileDirs.getValidatedDataPath()) / "Aufgaben"

    def _on_event(self, event: dict) -> None:
        if event["type"] == "node":   # one per node: thin it out for the socket
            now = time.monotonic()
            if now - self._last_node_event < NODE_EVENT_PERIOD_S:
                return
            self._last_node_event = now
        if event["type"] == "state" and event["state"] not in ACTIVE:
            try:
                self._host.end()
            except Exception as e:
                self._logger.warning(f"Could not restore the camera mode: {e}")
        self.sigAufgabenEvent.emit(event)

    def _active_run(self) -> Run:
        if self._run is None or self._run.state not in ACTIVE:
            raise HTTPException(status_code=409, detail="No job is running")
        return self._run

    def isAufgabenRunning(self) -> bool:
        return self._run is not None and self._run.state in ACTIVE

    # -- catalogue and documents -----------------------------------------------
    @APIExport()
    def getAufgabenCatalog(self) -> list:
        """Control nodes and the tasks this microscope can run, with JSON schemas.

        Device parameters list this setup's devices as ``enum``. Units are in
        each property's ``x-unit``; ``flags.ask_first`` marks steps a run must
        confirm.
        """
        providers = {kind: (lambda k=kind: self._host.devices(k)) for kind in DEVICE_KINDS}
        return aufgaben.catalogue(self._host.capabilities(), providers)

    @APIExport()
    def getAufgabenJobSchema(self) -> dict:
        """JSON Schema of a saved job."""
        return aufgaben.job_json_schema()

    @APIExport()
    def listAufgabenJobs(self) -> list:
        """Saved jobs and the bundled examples: name, source, file."""
        saved = [{"name": p.stem, "source": "saved", "file": str(p)}
                 for p in sorted(self._jobs_dir.glob("*.json"))]
        examples = [{"name": p.stem, "source": "example", "file": str(p)}
                    for p in sorted(aufgaben.EXAMPLES_DIR.glob("*.json"))]
        return saved + examples

    @APIExport()
    def getAufgabenJob(self, name: str) -> dict:
        """A saved job (or bundled example) by file name, without .json."""
        for folder in (self._jobs_dir, aufgaben.EXAMPLES_DIR):
            path = folder / f"{_safe(name)}.json"
            if path.is_file():
                return json.loads(path.read_text())
        raise HTTPException(status_code=404, detail=f"No job {name!r}")

    @APIExport(requestType="POST")
    def saveAufgabenJob(self, job: Dict[str, Any]) -> dict:
        """Save a job under its name; overwrites a saved job of the same name."""
        model = self._job(job)
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        path = self._jobs_dir / f"{_safe(model.name)}.json"
        path.write_text(model.to_json(indent=1))
        return {"name": path.stem, "file": str(path)}

    @APIExport()
    def deleteAufgabenJob(self, name: str) -> dict:
        """Delete a saved job (examples cannot be deleted)."""
        path = self._jobs_dir / f"{_safe(name)}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No saved job {name!r}")
        path.unlink()
        return {"deleted": path.stem}

    @APIExport(requestType="POST")
    def validateAufgabenJob(self, job: Dict[str, Any]) -> dict:
        """Errors, warnings, ask-first steps and whether the job moves or emits light."""
        return aufgaben.validate(self._job(job), capabilities=self._host.capabilities()).as_dict()

    @APIExport()
    def getAufgabenWellPlate(self, load_name: str, wells: str = "", offset_x_um: float = 0.0,
                             offset_y_um: float = 0.0) -> dict:
        """A useq well-plate plan for ImSwitch labware, in stage µm.

        Args:
            load_name: Labware name (getLabwareList); it must have a regular well grid.
            wells: Comma-separated well names (A1,B3); empty selects every well.
            offset_x_um: Stage X of the labware origin, as in getLabwareDefinition.
            offset_y_um: Stage Y of the labware origin.

        Returns the plan as JSON, ready for a job's well_plate resource. Rows run
        towards -Y and columns towards +X in both ImSwitch labware and useq.
        """
        import useq
        manager = self._labware()
        if manager is None or load_name not in self._plates:
            raise HTTPException(status_code=404,
                                detail=f"No labware {load_name!r} with a regular well grid")
        lab = manager.get_with_offset(load_name, offset_x_um, offset_y_um)
        names = [w.strip() for w in wells.split(",") if w.strip()] or lab.well_names_flat
        unknown = [w for w in names if w not in lab.wells]
        if unknown:
            raise HTTPException(status_code=422, detail=f"Unknown wells: {unknown}")
        rows = [lab.rows.index(lab.wells[w].row) for w in names]
        cols = [lab.columns.index(lab.wells[w].column) for w in names]
        a1 = lab.wells[f"{lab.rows[0]}{lab.columns[0]}"]
        plan = useq.WellPlatePlan(plate=load_name, a1_center_xy=(a1.x, a1.y),
                                  selected_wells=(rows, cols))
        return plan.model_dump(mode="json")

    @APIExport(requestType="POST")
    def importAufgabenUseq(self, sequence: Dict[str, Any]) -> dict:
        """A job from a useq-schema MDASequence (nested loops when possible)."""
        import useq
        return json.loads(aufgaben.job_from_sequence(useq.MDASequence(**sequence)).to_json())

    @APIExport(requestType="POST")
    def exportAufgabenUseq(self, job: Dict[str, Any]) -> dict:
        """The useq-schema MDASequence of a job without control flow."""
        try:
            seq = aufgaben.sequence_from_job(self._job(job))
        except aufgaben.NotExportable as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return seq.model_dump(mode="json", exclude_unset=True)

    # -- runs -----------------------------------------------------------------
    @APIExport(requestType="POST")
    def startAufgabenJob(self, request: StartAufgabenRequest) -> dict:
        """Validate and start a job. Refused while anything else drives the microscope.

        Steps flagged ask-first (homing, objective change) need ``confirmed``.
        Returns the run id and folder; follow it with getAufgabenStatus or
        sigAufgabenEvent.
        """
        job = self._job(request.job)
        with self._lock:
            if self.isAufgabenRunning():
                raise HTTPException(status_code=409, detail="A job is already running")
            others = busy_reasons(self._master, self._logger, ignore=("Aufgaben",))
            if others:
                raise HTTPException(status_code=409, detail=f"Refused: {', '.join(others)}")
            folder = self._runs_dir() / f"{datetime.now():%Y%m%d_%H%M%S}_{_safe(job.name)}"
            run = Run(job, self._host, parameters=request.parameters,
                      on_event=self._on_event, run_dir=folder)
            try:
                run.prepare(request.confirmed)
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            run.sink = OmeStore(folder, formats=job.storage.formats, layout=job.storage.layout,
                                stitch=job.storage.stitching if job.storage.stitched_preview
                                else None)
            (folder / "job.json").write_text(job.to_json(indent=1))
            self._host.begin()
            self._run = run
            run.start(request.confirmed)
        return {"run_id": run.run_id, "folder": str(folder)}

    @APIExport()
    def pauseAufgabenJob(self) -> dict:
        """Pause before the next step."""
        self._active_run().pause()
        return self.getAufgabenStatus()

    @APIExport()
    def resumeAufgabenJob(self) -> dict:
        self._active_run().resume()
        return self.getAufgabenStatus()

    @APIExport()
    def finishAufgabenJob(self) -> dict:
        """Stop before the next step and keep everything written so far."""
        self._active_run().finish()
        return self.getAufgabenStatus()

    @APIExport()
    def abortAufgabenJob(self, force: bool = False) -> dict:
        """Stop as aborted; ``force`` also stops the stage and switches all light off."""
        self._active_run().abort(force=force)
        return self.getAufgabenStatus()

    @APIExport()
    def getAufgabenStatus(self) -> dict:
        """State, current node, loop path, frames, pending question, variables."""
        return self._run.status() if self._run else {"state": "idle"}

    @APIExport()
    def answerAufgabenQuestion(self, answer: str) -> dict:
        """Answer the question the running job is waiting for."""
        run = self._active_run()
        if run.question is None:
            raise HTTPException(status_code=409, detail="The job is not asking anything")
        run.answer(answer)
        return {"answered": answer}

    @APIExport()
    def listAufgabenRuns(self, limit: int = 50) -> list:
        """Recent runs: folder, job name, final state, frames."""
        runs = []
        for folder in sorted(self._runs_dir().glob("*/run.jsonl"), reverse=True)[:limit]:
            lines = folder.read_text().splitlines()
            first, last = json.loads(lines[0]), json.loads(lines[-1])
            runs.append({"folder": str(folder.parent), "job": first.get("job", {}).get("name"),
                         "state": last.get("state", "running"), "frames": last.get("frames")})
        return runs
