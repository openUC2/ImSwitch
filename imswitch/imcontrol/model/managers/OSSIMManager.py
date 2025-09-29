# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any
import threading
from imswitch.imcommon.framework import Signal
from .ossim_client import OSSIMClient  # 你的客户端就放在同目录

class OSSIMManager:
    """Talks to your FastAPI DMD server."""
    def __init__(self, host: str, port: int, default_display_time: Optional[float] = None):
        self._client = OSSIMClient(URL=host, PORT=port)
        self._display_time = default_display_time
        self.sigStatus = Signal(dict)
        self._poll = False
        self._poll_thread: Optional[threading.Thread] = None

    def health(self) -> Dict[str, Any]:
        return self._client.health() or {"status": "error"}

    def list_patterns(self):
        return self._client.list_patterns()

    def set_pause(self, seconds: float):
        self._display_time = float(seconds)
        self._client.set_pause(self._display_time)
        return {"status": "ok", "display_time": self._display_time}

    def display(self, i: int):
        return self._client.display_pattern(int(i))

    def start(self, cycles: int = -1):
        if self._display_time is not None:
            self._client.set_pause(self._display_time)
        return self._client.start_viewer() if cycles == -1 else self._client.start_viewer_single_loop(int(cycles))

    def stop(self):
        return self._client.stop_loop()

    # optional: background status polling (call from controller/widget if needed)
    def start_poll(self, interval_s: float = 0.5):
        if self._poll: return
        self._poll = True
        def _loop():
            while self._poll:
                st = self._client.status()
                if isinstance(st, dict):
                    self.sigStatus.emit(st)
                threading.Event().wait(interval_s)
        self._poll_thread = threading.Thread(target=_loop, daemon=True); self._poll_thread.start()

    def stop_poll(self):
        self._poll = False
