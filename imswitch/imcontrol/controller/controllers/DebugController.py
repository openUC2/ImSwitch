"""
Debug Controller - hot-plug and run arbitrary Python code via the REST API.

This controller lets developers drop arbitrary Python code into a script
file (or send it directly in a request body) and execute it inside the
running ImSwitch process. The executed code gets a ``ctx`` object that
exposes the MasterController (all managers), the CommunicationChannel, the
setup info and a ``getController`` helper, so it runs with full access to the
live hardware/software stack and shares the same process/event loop.

Key properties:
- Scripts run in a background thread, so a long-running or blocking script
  does not freeze the FastAPI event loop.
- Every execution is wrapped in a try/except. Any error (including syntax
  errors) is captured, logged and returned to the caller -- ImSwitch itself
  never crashes because of a faulty debug script.
- Scripts are loaded from real files on disk via importlib, so breakpoints
  set in the file (e.g. in VS Code) map correctly and can be debugged
  natively while ImSwitch is running.

Safety is intentionally relaxed here: this is a developer debugging tool, not
a sandbox. Do not expose it on untrusted networks.
"""

import io
import os
import sys
import time
import uuid
import asyncio
import importlib.util
import threading
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import HTTPException

from imswitch.imcommon.model import APIExport, initLogger, dirtools
from ..basecontrollers import ImConWidgetController


# --------------------------------------------------------------------------- #
# Request models                                                              #
# --------------------------------------------------------------------------- #
class SaveScriptRequest(BaseModel):
    """Request body for saving/updating a debug script."""
    filename: str
    content: str


class CreateScriptRequest(BaseModel):
    """Request body for creating a new debug script from the template."""
    filename: str
    overwrite: bool = False


class RunScriptRequest(BaseModel):
    """Request body for running a script that lives on disk."""
    filename: str
    entrypoint: str = "run"
    background: bool = True
    # Optional timeout (seconds) used when background=False to bound the wait.
    timeout: float = 60.0


class RunCodeRequest(BaseModel):
    """Request body for running arbitrary code sent directly in the request."""
    code: str
    entrypoint: str = "run"
    background: bool = True
    timeout: float = 60.0
    # Optional human-readable name for the job.
    name: Optional[str] = None


class JobRequest(BaseModel):
    """Request body referencing an existing job."""
    jobId: str


# --------------------------------------------------------------------------- #
# Script execution context                                                    #
# --------------------------------------------------------------------------- #
class ScriptContext:
    """Object handed to every debug script as ``ctx``.

    It provides live access to the running ImSwitch instance. Scripts should
    cooperatively check ``ctx.stop_requested()`` in long loops so they can be
    stopped via the API.
    """

    def __init__(self, controller: "DebugController", stop_event: threading.Event,
                 job_id: str):
        self._controller = controller
        self._stop_event = stop_event
        self.jobId = job_id
        # Direct access to the live stack.
        self.master = controller._master
        self.commChannel = controller._commChannel
        self.setupInfo = controller._setupInfo
        self.logger = controller._logger

    def getController(self, name: str):
        """Return another registered controller by name (or None)."""
        return self._controller._master.getController(name)

    def stop_requested(self) -> bool:
        """Return True once a stop has been requested for this job."""
        return self._stop_event.is_set()

    def sleep(self, seconds: float, poll: float = 0.1) -> None:
        """Stop-aware sleep. Returns early if a stop is requested."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_event.is_set():
                return
            time.sleep(min(poll, max(0.0, deadline - time.time())))

    def run_coroutine(self, coro, timeout: Optional[float] = None):
        """Schedule a coroutine on ImSwitch's main asyncio loop and wait.

        Use this when a script needs to call async API methods that must run
        on the server event loop rather than in the worker thread.
        """
        loop = self._controller._loop
        if loop is None:
            raise RuntimeError("No asyncio event loop captured yet.")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout)


# --------------------------------------------------------------------------- #
# Controller                                                                   #
# --------------------------------------------------------------------------- #
class DebugController(ImConWidgetController):
    """Controller exposing endpoints to hot-plug and run Python code."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # Directory that holds editable debug scripts.
        self._scriptsDir = os.path.join(dirtools.UserFileDirs.Root, "debug_scripts")
        os.makedirs(self._scriptsDir, exist_ok=True)

        # Registry of jobs: jobId -> job state dict.
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobsLock = threading.Lock()

        # Reference to the server asyncio loop, captured lazily from endpoints.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Make sure the template (and a default scratch script) exist on disk.
        self._ensureTemplate()
        self._ensureScratch()

        self._logger.info(f"DebugController ready. Scripts dir: {self._scriptsDir}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _captureLoop(self) -> None:
        """Capture the running asyncio loop so worker threads can use it."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # Not called from within a loop; ignore.
                pass

    def _safeName(self, filename: str) -> str:
        """Sanitize a filename to stay inside the scripts directory."""
        name = os.path.basename(filename.strip())
        if not name:
            raise HTTPException(status_code=400, detail="Empty filename.")
        if not name.endswith(".py"):
            name += ".py"
        return name

    def _scriptPath(self, filename: str) -> str:
        return os.path.join(self._scriptsDir, self._safeName(filename))

    def _ensureTemplate(self) -> None:
        path = os.path.join(self._scriptsDir, "debug_template.py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_TEMPLATE_CONTENT)

    def _ensureScratch(self) -> None:
        path = os.path.join(self._scriptsDir, "scratch.py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_TEMPLATE_CONTENT)

    def _newJob(self, name: str) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "jobId": job_id,
            "name": name,
            "status": "queued",      # queued | running | finished | error | stopped
            "startTime": None,
            "endTime": None,
            "result": None,
            "error": None,
            "traceback": None,
            "stdout": "",
            "_thread": None,
            "_stopEvent": threading.Event(),
        }
        with self._jobsLock:
            self._jobs[job_id] = job
        return job

    def _jobPublic(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable view of a job (without internals)."""
        return {
            "jobId": job["jobId"],
            "name": job["name"],
            "status": job["status"],
            "startTime": job["startTime"],
            "endTime": job["endTime"],
            "result": job["result"],
            "error": job["error"],
            "traceback": job["traceback"],
            "stdout": job["stdout"],
        }

    def _loadModuleFromFile(self, path: str):
        """Load a fresh module from a file path for native debugging."""
        mod_name = f"imswitch_debug_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        # Register so tracebacks/breakpoints resolve correctly.
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module, mod_name

    @staticmethod
    def _jsonSafe(value: Any) -> Any:
        """Best-effort conversion of a return value into something JSON-safe."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): DebugController._jsonSafe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DebugController._jsonSafe(v) for v in value]
        return repr(value)

    def _runJob(self, job: Dict[str, Any], source: str, *, is_file: bool,
                entrypoint: str) -> None:
        """Worker body that actually executes the user code with isolation."""
        ctx = ScriptContext(self, job["_stopEvent"], job["jobId"])
        buffer = io.StringIO()
        job["status"] = "running"
        job["startTime"] = datetime.now().isoformat()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                if is_file:
                    module, mod_name = self._loadModuleFromFile(source)
                    try:
                        entry = getattr(module, entrypoint, None)
                        if entry is None:
                            raise AttributeError(
                                f"Script has no '{entrypoint}(ctx)' function."
                            )
                        result = entry(ctx)
                    finally:
                        sys.modules.pop(mod_name, None)
                else:
                    namespace: Dict[str, Any] = {"__name__": "imswitch_debug"}
                    compiled = compile(source, "<imswitch-debug>", "exec")
                    exec(compiled, namespace)  # noqa: S102 - intentional dev tool
                    entry = namespace.get(entrypoint)
                    if callable(entry):
                        result = entry(ctx)
                    else:
                        # No entrypoint: treat the module body as the script.
                        result = namespace.get("result")
            job["result"] = self._jsonSafe(result)
            job["status"] = "stopped" if job["_stopEvent"].is_set() else "finished"
        except Exception as exc:  # noqa: BLE001 - we want to catch everything
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["traceback"] = traceback.format_exc()
            self._logger.error(
                f"Debug job '{job['name']}' ({job['jobId']}) failed:\n"
                f"{job['traceback']}"
            )
        finally:
            job["stdout"] = buffer.getvalue()
            job["endTime"] = datetime.now().isoformat()

    def _startJob(self, job: Dict[str, Any], source: str, *, is_file: bool,
                  entrypoint: str, background: bool, timeout: float) -> Dict[str, Any]:
        thread = threading.Thread(
            target=self._runJob,
            args=(job, source),
            kwargs={"is_file": is_file, "entrypoint": entrypoint},
            name=f"DebugJob-{job['jobId']}",
            daemon=True,
        )
        job["_thread"] = thread
        thread.start()
        if not background:
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Still running after the timeout -> report as background job.
                return {
                    **self._jobPublic(job),
                    "note": "Still running after timeout; query getJobStatus.",
                }
        return self._jobPublic(job)

    # ------------------------------------------------------------------ #
    # API: script file management                                        #
    # ------------------------------------------------------------------ #
    @APIExport(runOnUIThread=False)
    def listScripts(self) -> Dict[str, Any]:
        """List all debug scripts available on disk."""
        self._captureLoop()
        files = sorted(
            f for f in os.listdir(self._scriptsDir) if f.endswith(".py")
        )
        return {"scriptsDir": self._scriptsDir, "scripts": files}

    @APIExport(runOnUIThread=False)
    def getScript(self, filename: str) -> Dict[str, Any]:
        """Return the content of a debug script."""
        self._captureLoop()
        path = self._scriptPath(filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Script not found: {filename}")
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        return {"filename": os.path.basename(path), "content": content}

    @APIExport(runOnUIThread=False)
    def getTemplate(self) -> Dict[str, str]:
        """Return the debug-script template content."""
        self._captureLoop()
        return {"content": _TEMPLATE_CONTENT}

    @APIExport(runOnUIThread=False, requestType="POST")
    def saveScript(self, request: SaveScriptRequest) -> Dict[str, Any]:
        """Create or overwrite a debug script on disk."""
        self._captureLoop()
        path = self._scriptPath(request.filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(request.content)
        return {"success": True, "filename": os.path.basename(path), "path": path}

    @APIExport(runOnUIThread=False, requestType="POST")
    def createScript(self, request: CreateScriptRequest) -> Dict[str, Any]:
        """Create a new debug script pre-filled with the template."""
        self._captureLoop()
        path = self._scriptPath(request.filename)
        if os.path.exists(path) and not request.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"Script already exists: {os.path.basename(path)}",
            )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_TEMPLATE_CONTENT)
        return {"success": True, "filename": os.path.basename(path), "path": path}

    @APIExport(runOnUIThread=False, requestType="POST")
    def deleteScript(self, request: JobRequest) -> Dict[str, Any]:
        """Delete a debug script (the ``jobId`` field carries the filename)."""
        self._captureLoop()
        path = self._scriptPath(request.jobId)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Script not found.")
        os.remove(path)
        return {"success": True, "filename": os.path.basename(path)}

    # ------------------------------------------------------------------ #
    # API: execution                                                     #
    # ------------------------------------------------------------------ #
    @APIExport(runOnUIThread=False, requestType="POST")
    def runScript(self, request: RunScriptRequest) -> Dict[str, Any]:
        """Hot-load a script file and run its entrypoint in a thread.

        The file is (re)loaded from disk on every call, so edits take effect
        immediately and breakpoints inside the file can be hit natively.
        """
        self._captureLoop()
        path = self._scriptPath(request.filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Script not found: {request.filename}")
        job = self._newJob(name=os.path.basename(path))
        return self._startJob(
            job, path, is_file=True, entrypoint=request.entrypoint,
            background=request.background, timeout=request.timeout,
        )

    @APIExport(runOnUIThread=False, requestType="POST")
    def runCode(self, request: RunCodeRequest) -> Dict[str, Any]:
        """Execute arbitrary Python code sent in the request body.

        If the code defines the entrypoint function (default ``run``), it is
        called with the ``ctx`` object. Otherwise the module body itself is
        executed and an optional ``result`` variable is returned.
        """
        self._captureLoop()
        job = self._newJob(name=request.name or "inline-code")
        return self._startJob(
            job, request.code, is_file=False, entrypoint=request.entrypoint,
            background=request.background, timeout=request.timeout,
        )

    # ------------------------------------------------------------------ #
    # API: job inspection / control                                      #
    # ------------------------------------------------------------------ #
    @APIExport(runOnUIThread=False)
    def listJobs(self) -> Dict[str, Any]:
        """List all known jobs and their current status."""
        self._captureLoop()
        with self._jobsLock:
            jobs = [self._jobPublic(j) for j in self._jobs.values()]
        return {"jobs": jobs}

    @APIExport(runOnUIThread=False)
    def getJobStatus(self, jobId: str) -> Dict[str, Any]:
        """Return the status, result, stdout and any error for a job."""
        self._captureLoop()
        with self._jobsLock:
            job = self._jobs.get(jobId)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {jobId}")
        return self._jobPublic(job)

    @APIExport(runOnUIThread=False, requestType="POST")
    def stopJob(self, request: JobRequest) -> Dict[str, Any]:
        """Cooperatively request a running job to stop.

        The script must check ``ctx.stop_requested()`` for this to take effect.
        """
        self._captureLoop()
        with self._jobsLock:
            job = self._jobs.get(request.jobId)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {request.jobId}")
        job["_stopEvent"].set()
        return {"success": True, "jobId": request.jobId, "status": job["status"]}

    @APIExport(runOnUIThread=False, requestType="POST")
    def clearJobs(self) -> Dict[str, Any]:
        """Remove finished/error/stopped jobs from the registry."""
        self._captureLoop()
        with self._jobsLock:
            keep = {
                jid: j for jid, j in self._jobs.items()
                if j["status"] in ("queued", "running")
            }
            removed = len(self._jobs) - len(keep)
            self._jobs = keep
        return {"success": True, "removed": removed}


# --------------------------------------------------------------------------- #
# Template content written to disk for new scripts                            #
# --------------------------------------------------------------------------- #
_TEMPLATE_CONTENT = '''"""
ImSwitch debug script template.

Define a function `run(ctx)`. It is called when you trigger this script via
the REST API (DebugController/runScript). The `ctx` object gives you full
access to the live ImSwitch instance:

    ctx.master              -> MasterController (all *Manager instances)
    ctx.commChannel         -> CommunicationChannel
    ctx.setupInfo           -> active setup info
    ctx.logger              -> logger (use instead of print)
    ctx.getController(name) -> another controller, e.g. "Experiment"
    ctx.stop_requested()    -> True once a stop was requested via the API
    ctx.sleep(seconds)      -> stop-aware sleep
    ctx.run_coroutine(coro) -> run an async call on the server event loop

The script runs in a background thread and shares the process, so you have
access to all managers and controllers. Any exception is caught and reported
back through the API instead of crashing ImSwitch.

You can set breakpoints in this file and debug it natively while ImSwitch is
running, since it is hot-loaded from disk on every run.
"""


def run(ctx):
    ctx.logger.info("Debug script started")

    # Example: list available hardware
    detectors = ctx.master.detectorsManager.getAllDeviceNames()
    positioners = ctx.master.positionersManager.getAllDeviceNames()
    lasers = ctx.master.lasersManager.getAllDeviceNames()

    ctx.logger.info(f"Detectors:   {detectors}")
    ctx.logger.info(f"Positioners: {positioners}")
    ctx.logger.info(f"Lasers:      {lasers}")

    # Example: cooperative long-running loop that can be stopped via the API
    # for i in range(100):
    #     if ctx.stop_requested():
    #         ctx.logger.info("Stop requested, exiting loop")
    #         break
    #     ctx.logger.info(f"working... {i}")
    #     ctx.sleep(1.0)

    # The returned value is reported back through getJobStatus (JSON-safe).
    return {
        "detectors": detectors,
        "positioners": positioners,
        "lasers": lasers,
    }
'''
