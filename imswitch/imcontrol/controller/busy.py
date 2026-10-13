"""What is driving the instrument right now.

One check for everything that must not start while something else runs: an
Aufgaben job, an experiment, a recording, a workflow, the stage map or its
prescan, a focus-lock calibration, Arkitekt actions, an unattended firmware
update. A check that fails is logged and does not block.
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
}


def busy_reasons(master, logger=None, ignore=()) -> list:
    """Lower-case reasons, e.g. ["an experiment is running"]; empty when idle.

    ``ignore`` names controllers not to check (a controller asking about itself).
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
    return reasons
