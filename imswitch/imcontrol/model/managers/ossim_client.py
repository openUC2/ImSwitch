# ossim_client.py
import time
from typing import Any, Dict, List, Optional, Union

import requests


class OSSIMClient:
    """
    A drop-in style client for the DMD FastAPI server you provided.
    Method names mirror the SIMClient used in ImSwitch for easier integration.
    
    Key mappings:
      - start_viewer()                -> POST /start  (cycles=-1)
      - start_viewer_single_loop(n)   -> POST /start  (cycles=n)
      - wait_for_viewer_completion()  -> poll GET /status until not running / cycles done
      - set_pause(period)             -> set display_time; if running, restart loop
      - stop_loop()                   -> POST /stop
      - display_pattern(i)            -> GET /display/{i}
      
    Extra helpers:
      - health()                      -> GET /health
      - status()                      -> GET /status
      - list_patterns()               -> GET /patterns
      - reload_patterns(dir, files)   -> POST /reload (json)
    """

    def __init__(
        self,
        URL: str,
        PORT: Union[int, str] = 8000,
        *,
        default_timeout: float = 1.0,
        poll_interval: float = 0.2,
    ):
        self.base_url = f"http://{URL}:{PORT}"
        self.default_timeout = float(default_timeout)
        self.poll_interval = float(poll_interval)
        self.session = requests.Session()

        # Keep same style as SIMClient (key names only for readability)
        self.commands = {
            "health": "/health",
            "status": "/status",
            "patterns": "/patterns",
            "display_pattern": "/display/",    # + {pattern_id}
            "start": "/start",
            "stop_loop": "/stop",
            "reload": "/reload",
        }

        # Client-side state
        self._display_time: Optional[float] = None  # seconds; None -> use server default

    # -------------- low-level HTTP helpers --------------
    def _get(self, path: str, *, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
        try:
            r = self.session.get(self.base_url + path, params=params, timeout=timeout or self.default_timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[OSSIMClient][GET {path}] {e}")
            return -1

    def _post(
        self,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        try:
            r = self.session.post(self.base_url + path, params=params, json=json, timeout=timeout or self.default_timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[OSSIMClient][POST {path}] {e}")
            return -1

    # -------------- ImSwitch-compatible methods --------------
    def start_viewer(self) -> Union[int, Dict[str, Any]]:
        """
        Start infinite pattern cycling (cycles = -1).
        Uses self._display_time if set via set_pause(), otherwise server default.
        """
        params = {"cycles": -1}
        if self._display_time is not None:
            params["display_time"] = float(self._display_time)

        resp = self._post(self.commands["start"], params=params)
        # If already running and user just changed pause, try to restart with new pause
        if isinstance(resp, dict) and resp.get("status") == "error" and "Already running" in str(resp.get("message", "")):
            # Graceful restart with new display_time
            self.stop_loop()
            resp = self._post(self.commands["start"], params=params)
        return resp

    def start_viewer_single_loop(self, number_of_runs: int, timeout: float = 2.0) -> Union[int, Dict[str, Any]]:
        """
        Start a finite number of cycles (cycles = number_of_runs).
        Uses self._display_time if set via set_pause(), otherwise server default.
        """
        params = {"cycles": int(number_of_runs)}
        if self._display_time is not None:
            params["display_time"] = float(self._display_time)
        return self._post(self.commands["start"], params=params, timeout=timeout)

    def wait_for_viewer_completion(self, *, timeout: Optional[float] = None) -> None:
        """
        Block until the server reports not running OR the finite cycles have completed.
        For infinite loops, this will block until stop_loop() is called or timeout is reached.
        """
        start_t = time.time()
        while True:
            st = self.status()
            if st == -1:
                # transient error; keep trying unless timed out
                if timeout is not None and (time.time() - start_t) > timeout:
                    print("[OSSIMClient] wait_for_viewer_completion timed out (status fetch errors).")
                    return
                time.sleep(self.poll_interval)
                continue

            running = bool(st.get("running", False))
            max_cycles = int(st.get("max_cycles", -1))
            cur_cycle = int(st.get("current_cycle", 0))

            if not running:
                return

            # If a finite run is requested and completed
            if max_cycles > 0 and cur_cycle >= max_cycles:
                return

            if timeout is not None and (time.time() - start_t) > timeout:
                print("[OSSIMClient] wait_for_viewer_completion timed out.")
                return

            time.sleep(self.poll_interval)

    def set_pause(self, period: float) -> None:
        """
        Set the per-pattern display time (seconds). If a loop is running, restart with new pause.
        """
        self._display_time = float(period)
        st = self.status()
        if st != -1 and bool(st.get("running", False)):
            # Restart with same cycles if finite; otherwise infinite
            max_cycles = int(st.get("max_cycles", -1))
            self.stop_loop()
            if max_cycles > 0:
                self.start_viewer_single_loop(max_cycles)
            else:
                self.start_viewer()

    def stop_loop(self) -> Union[int, Dict[str, Any]]:
        return self._post(self.commands["stop_loop"])

    def display_pattern(self, iPattern: int) -> Union[int, Dict[str, Any]]:
        """
        Display a single pattern immediately. This does not start cycling.
        """
        return self._get(self.commands["display_pattern"] + str(int(iPattern)))

    # -------------- extra helpers --------------
    def health(self) -> Union[int, Dict[str, Any]]:
        return self._get(self.commands["health"])

    def status(self) -> Union[int, Dict[str, Any]]:
        return self._get(self.commands["status"])

    def list_patterns(self) -> Union[int, Dict[str, Any]]:
        return self._get(self.commands["patterns"])

    def reload_patterns(
        self,
        pattern_dir: Optional[str] = None,
        pattern_files: Optional[List[str]] = None,
    ) -> Union[int, Dict[str, Any]]:
        """
        Reload patterns on the server. Provide either directory, file list, or both.
        """
        payload: Dict[str, Any] = {}
        if pattern_dir is not None:
            payload["pattern_dir"] = pattern_dir
        if pattern_files is not None:
            payload["pattern_files"] = list(pattern_files)
        return self._post(self.commands["reload"], json=payload or {})


# ----------------------- usage example -----------------------
if __name__ == "__main__":
    # Example IP/PORT; change to your server
    client = OSSIMClient(URL="169.254.165.4", PORT=8000)

    print("Health:", client.health())
    print("Patterns:", client.list_patterns())

    # Show a single pattern
    client.display_pattern(0)

    # Infinite loop with 0.5 s per pattern
    client.set_pause(0.5)
    client.start_viewer()

    # Wait a few cycles then stop
    time.sleep(3)
    print("Status:", client.status())
    client.stop_loop()

    # Run exactly 5 cycles with 0.2 s per pattern
    client.set_pause(0.2)
    client.start_viewer_single_loop(5)
    client.wait_for_viewer_completion()
    print("Done single run. Status:", client.status())
