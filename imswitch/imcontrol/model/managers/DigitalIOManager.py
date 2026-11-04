"""Named digital lines of the UC2 board.

The ``digitalIO`` setup block names the board's outputs and inputs
(``{"outputs": {"shutter": 1}, "inputs": {"door": 2}, "invert": ["door"]}``),
so jobs and endpoints say ``shutter`` instead of line 1, and active-low lines
read and write as logical levels. Without the block the raw lines "1".."3"
are offered whenever a board is connected.
"""
import time
from typing import Dict, List, Optional

RAW_LINES = ("1", "2", "3")
POLL_S = 0.02


class DigitalIOManager:
    def __init__(self, digitalIOInfo=None, **lowLevelManagers):
        outputs = dict(getattr(digitalIOInfo, "outputs", None) or {})
        inputs = dict(getattr(digitalIOInfo, "inputs", None) or {})
        named = bool(outputs or inputs)
        self._outputs: Dict[str, int] = {k: int(v) for k, v in outputs.items()} if named \
            else {line: int(line) for line in RAW_LINES}
        self._inputs: Dict[str, int] = {k: int(v) for k, v in inputs.items()} if named \
            else {line: int(line) for line in RAW_LINES}
        self._invert = set(getattr(digitalIOInfo, "invert", None) or [])
        device = getattr(digitalIOInfo, "rs232device", None) or "ESP32"
        self._board = None
        try:
            self._board = lowLevelManagers["rs232sManager"][device]._esp32
        except Exception:
            pass   # no board: no digital I/O

    @property
    def available(self) -> bool:
        return self._board is not None

    @property
    def outputs(self) -> List[str]:
        return list(self._outputs) if self.available else []

    @property
    def inputs(self) -> List[str]:
        return list(self._inputs) if self.available else []

    def _line(self, table: Dict[str, int], name: str, kind: str) -> int:
        if not self.available:
            raise RuntimeError("no UC2 board connected")
        if str(name) not in table:
            raise ValueError(f"no digital {kind} {name!r}; this setup has {list(table)}")
        return table[str(name)]

    def set_output(self, name: str, high: bool) -> None:
        line = self._line(self._outputs, name, "output")
        level = bool(high) != (name in self._invert)
        self._board.digitalout.set_digitalout(digitaloutid=line, digitaloutval=int(level))

    def pulse_output(self, name: str, duration_ms: float) -> None:
        # The firmware pulse has its own fixed length; longer pulses are timed here.
        self.set_output(name, True)
        time.sleep(duration_ms / 1000.0)
        self.set_output(name, False)

    def read_input(self, name: str) -> bool:
        line = self._line(self._inputs, name, "input")
        reply = self._board.digitalin.get_digitalin(digitalinid=line)
        reply = reply[0] if isinstance(reply, list) else reply
        return bool(reply["digitalin"]["digitalinval"]) != (name in self._invert)

    def wait_for_input(self, name: str, edge: str, timeout_s: Optional[float] = None) -> bool:
        """Wait for "rising"/"falling" (an edge needs the opposite level first) or
        "high"/"low"; False on timeout."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        want = edge in ("rising", "high")
        for level in ((not want, want) if edge in ("rising", "falling") else (want,)):
            while self.read_input(name) != level:
                if deadline is not None and time.monotonic() > deadline:
                    return False
                time.sleep(POLL_S)
        return True
