"""Run control of the flat step engine (WorkflowManager).

Pause used to be a flag nothing read, resume raised AttributeError, and waits
were plain time.sleep calls that ignored stop. These tests pin the fixed
behaviour without hardware.
"""
import threading
import time

from imswitch.imcontrol.model.managers.WorkflowManager import (
    Workflow, WorkflowContext, WorkflowStep, WorkflowsManager,
)


def _steps(n, log, delay=0.02):
    def work(i):
        time.sleep(delay)
        log.append(i)
    return [WorkflowStep(name=f"s{i}", main_func=work, main_params={"i": i}, step_id=str(i))
            for i in range(n)]


def _wait_until(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_pause_blocks_between_steps_and_resume_continues():
    log = []
    mgr = WorkflowsManager()
    ctx = WorkflowContext()
    mgr.start_workflow(Workflow(_steps(20, log), mgr), ctx)
    assert _wait_until(lambda: len(log) >= 2)
    mgr.pause_workflow()
    assert mgr.get_status()["status"] == "paused"
    time.sleep(0.1)                       # let the running step finish
    frozen = len(log)
    time.sleep(0.2)
    assert len(log) == frozen, "steps kept running while paused"
    assert mgr.resume_workflow() == {"status": "resumed"}
    assert _wait_until(lambda: len(log) == 20)
    assert log == list(range(20))


def test_stop_wakes_a_paused_workflow():
    log = []
    mgr = WorkflowsManager()
    ctx = WorkflowContext()
    _, thread = Workflow(_steps(50, log), mgr).run_in_background(ctx)
    mgr.current_context, mgr.current_thread = ctx, thread
    assert _wait_until(lambda: len(log) >= 1)
    ctx.request_pause()
    time.sleep(0.05)
    ctx.request_stop()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(log) < 50


def test_context_sleep_returns_early_on_stop():
    ctx = WorkflowContext()
    threading.Timer(0.1, ctx.request_stop).start()
    t0 = time.time()
    assert ctx.sleep(10.0) is False
    assert time.time() - t0 < 2.0


def test_context_sleep_full_duration_without_stop():
    ctx = WorkflowContext()
    t0 = time.time()
    assert ctx.sleep(0.05) is True
    assert time.time() - t0 >= 0.04


def test_flags_remain_assignable():
    # WorkflowStep sets context.should_stop = True on failure.
    ctx = WorkflowContext()
    ctx.should_stop = True
    assert ctx.should_stop and not ctx.should_pause
    ctx.should_stop = False
    ctx.should_pause = True
    assert ctx.should_pause
    ctx.should_pause = False
    assert not ctx.should_pause
