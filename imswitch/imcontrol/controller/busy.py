"""What is driving the instrument right now.

One check for everything that must not start while something else runs: an
Aufgaben job, an experiment, Arkitekt actions, an unattended firmware update.
A check that fails is logged and does not block.
"""
BUSY_STATES = ("running", "paused", "stopping")


def busy_reasons(master, logger=None, ignore=()) -> list:
    """Lower-case reasons, e.g. ["an experiment is running"]; empty when idle.

    ``ignore`` names controllers not to check (a controller asking about itself).
    """
    get = getattr(master, "getController", lambda name: None)
    checks = []
    experiment = get("Experiment") if "Experiment" not in ignore else None
    if experiment is not None:
        checks.append(("an experiment is running", lambda: (
            experiment.getExperimentStatus().get("status") in BUSY_STATES
            or getattr(experiment, "_experiment_starting", False))))
    recording = get("Recording") if "Recording" not in ignore else None
    if recording is not None:
        checks.append(("a recording is running", recording.isRecording))
    workflow = getattr(get("Workflow"), "workflow_manager", None)
    if workflow is not None and "Workflow" not in ignore:
        checks.append(("a workflow is running",
                       lambda: workflow.get_status().get("status") in BUSY_STATES))
    aufgaben = get("Aufgaben") if "Aufgaben" not in ignore else None
    if aufgaben is not None:
        checks.append(("an Aufgaben job is running", aufgaben.isAufgabenRunning))
    reasons = []
    for label, predicate in checks:
        try:
            if predicate():
                reasons.append(label)
        except Exception as e:
            if logger is not None:
                logger.warning(f"Busy check '{label}' failed (not blocking): {e}")
    return reasons
