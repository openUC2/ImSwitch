"""Exclusive use of the microscope by a lab-automation workcell (SiLA Lock Controller).

While a workcell holds the SiLA lock, the HTTP API (the ImSwitch UI, scripts) may still read and
stop, but not move, switch light or start anything: those requests get 423 Locked. Calls the
SiLA server makes on the workcell's behalf run with ``sila_call`` set and pass, also through
``busy_reasons``. The lock state comes from the SiLA server (``set_lock_state``); no server, no
lock.
"""
import contextvars

sila_call = contextvars.ContextVar("sila_call", default=False)
"""True in the SiLA server's thread and the worker threads it starts with asyncio.to_thread."""

# Endpoints that stay open while locked: reading, watching, stopping.
OPEN_PREFIXES = ("get", "is", "list", "return", "has", "read", "stop", "cancel", "abort",
                 "force", "mjpeg", "video", "download", "release")
OPEN_CONTROLLERS = ("SiLa2Controller",)

_lock_state = None  # callable -> str: who holds the lock ("" = nobody)


def set_lock_state(fn) -> None:
    """Install the function that names the lock holder ("" when unlocked)."""
    global _lock_state
    _lock_state = fn


def holder() -> str:
    """Who holds the lock, "" when nobody does or when called on the workcell's behalf."""
    if _lock_state is None or sila_call.get():
        return ""
    return _lock_state() or ""


def http_refusal(path: str) -> str:
    """Reason to refuse an HTTP request to ``path`` while locked; "" to let it through."""
    who = holder()
    if not who:
        return ""
    parts = [p for p in path.split("/") if p]
    if "api" not in parts or len(parts) < 2:
        return ""
    controller, method = parts[-2], parts[-1]
    if (not controller.endswith("Controller") or controller in OPEN_CONTROLLERS
            or method.startswith(OPEN_PREFIXES)):
        return ""
    return (f"Refused: a lab-automation workcell holds the microscope ({who}). It unlocks when "
            "the workcell is done; SiLa2Controller/releaseWorkcellLock frees it by hand.")
