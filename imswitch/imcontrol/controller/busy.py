"""What is driving the instrument right now.

One check for everything that must not start while something else runs: an
Aufgaben job, an experiment, a recording, a workflow, the stage map or its
prescan, a focus-lock calibration, an SLM sequence acquisition, Arkitekt
actions, an unattended firmware update, a deconvolution computing on the
Pi. A check that fails is logged and does not block.
"""
BUSY_STATES = ("running", "paused", "stopping")


def _experiment_busy(experiment) -> bool:
    return (experiment.getExperimentStatus().get("status") in BUSY_STATES
            or getattr(experiment, "_experiment_starting", False))


def _workflow_busy(workflow) -> bool:
    return workflow.workflow_manager.get_status().get("status") in BUSY_STATES


# controller name -> (reason, predicate on the controller)
CHECKS = {
    "Experiment": ("an experiment is running", _experiment_busy),
    "Recording": ("a recording is running", lambda c: c.isRecording()),
    "Workflow": ("a workflow is running", _workflow_busy),
    "Aufgaben": ("an Aufgaben job is running", lambda c: c.isAufgabenRunning()),
    "StageMap": ("the stage map or its prescan is running", lambda c: c.isStageMapBusy()),
    "FocusLock": ("a focus-lock calibration is running", lambda c: c.isCalibrationRunning()),
    "OPM": ("an OPM scan is running", lambda c: c.isOPMScanRunning()),
    "SLM": ("an SLM sequence acquisition or calibration is running", lambda c: c.isSLMBusy()),
    "FrameCalibration": ("a FRAME calibration is running", lambda c: c.isFrameCalibrationRunning()),
    "Processing": ("a deconvolution is computing on the Pi", lambda c: c.isProcessingBusy()),
}


def placeholder_cameras(master) -> list:
    """Acquisition cameras whose real camera did not open, so a simulated one stands in."""
    names = []
    try:
        manager = master.detectorsManager
        for name in manager.getAllDeviceNames():
            detector = manager[name]
            if detector.forAcquisition and detector.isPlaceholder is True:
                names.append(name)
    except Exception:   # no detectors manager (tests, partial setups): nothing to refuse
        pass
    return names


def camera_reasons(master) -> list:
    """Why frame-taking work must not start: its images would be fake test images."""
    return [f"camera {name} did not open and a simulated camera stands in (plug it in and "
            f"restart ImSwitch)" for name in placeholder_cameras(master)]


def busy_reasons(master, logger=None, ignore=(), camera=False) -> list:
    """Lower-case reasons, e.g. ["an experiment is running"]; empty when idle.

    ``ignore`` names controllers not to check (a controller asking about itself).
    ``camera=True`` (work that takes frames) also refuses a stand-in camera.
    """
    get = getattr(master, "getController", lambda name: None)
    reasons = []
    for name, (label, predicate) in CHECKS.items():
        controller = get(name) if name not in ignore else None
        if controller is None:
            continue
        try:
            if predicate(controller):
                reasons.append(label)
        except Exception as e:
            if logger is not None:
                logger.warning(f"Busy check '{label}' failed (not blocking): {e}")
    return reasons + (camera_reasons(master) if camera else [])
