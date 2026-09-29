"""
StepperXYStageManager
=====================
ImSwitch PositionerManager for a 2-axis (XY) stepper stage that speaks a
simple ASCII serial protocol at 9600 baud.

Serial protocol summary
-----------------------
All commands are terminated with ``\n``.  All replies end with ``OK\n`` on
success or ``ERR <reason>\n`` on failure.

  V sx sy             – set velocity: X = sx steps/s, Y = sy steps/s
  MOVE dx dy [speed]  – relative move in steps; optional per-move speed
  STOP                – stop immediately (device sends V 0 0 internally)
  POS                 – query step counts, reply "<x> <y>" (microsteps)
  BUSY                – reply "1" while a MOVE/SNAKE runs or motors step, else "0"
  POWER [x y]         – get/set per-axis peak duty 0..32767 (0 = firmware default)
  PING                – connectivity check
  HELP                – print help text
  MICROSTEP           – query the microstep mode actually in use
  MICROSTEP N         – store microstep N (active after device reboot)
  SNAKE nx ny sx sy speed pause [holdPct]
                      – execute a snake scan:
                        nx/ny    : number of steps in X/Y
                        sx/sy    : step sizes in steps
                        speed    : movement speed (steps/s)
                        pause    : pause between moves (ms)
                        holdPct  : optional hold-torque % during pauses

Configuration keys (managerProperties)
---------------------------------------
  rs232device        – key into rs232sManager (must be a SerialManager/RS232Manager)
  stepsizeX          – µm per step in X (default: 1.0); ignored if
                       umPerElectricalCycleX is set
  stepsizeY          – µm per step in Y (default: 1.0)
  umPerElectricalCycleX / Y
                     – µm travelled per electrical cycle (nominally 4 × trace
                       pitch).  If set, µm/step = this / MICROSTEP read from the
                       device at startup, so changing MICROSTEP cannot silently
                       break the scale.
  powerX / powerY    – per-axis peak duty sent at startup via POWER (0..32767).
                       Use to equalise current for unequal trace resistance.
  approachDirectionX / Y
                     – +1 / -1: every move ends travelling in this direction.
                       A move the other way goes past the target by
                       approachOvershootSteps first. Removes the friction
                       dead band on reversals (PCB stage: ~230 µm per
                       reversal). 0 = off (default). Replaces backlashX/Y.
  approachOvershootSteps – overshoot for the above (default 8 = one full step)
  settleMs           – wait after a blocking move has finished (default 0)
  maxSpeedSteps      – cap on the per-move speed in steps/s; ExperimentController
                       asks for 20000 (default: no cap, firmware limit 2000)
  firmware           – dict of firmware settings sent at startup, e.g.
                       {"ACCEL": 1500, "HOLD": 0, "DITHER": "1 16 50"};
                       "PHASE_X"/"PHASE_Y": list of per-µstep phase corrections
  initialSpeedX      – default speed in steps/s for X (default: 800)
  initialSpeedY      – default speed in steps/s for Y (default: 800)
  microsteps         – microstep divisor to write on startup (optional)
  swapXY             – if True, swap X and Y axes on the wire (default: False);
                       use when the stage is mounted 90° rotated
  backlashX          – backlash compensation in steps for X axis (default: 0);
                       positive value overshoots and returns when direction reverses
  backlashY          – backlash compensation in steps for Y axis (default: 0)
  idleStopTimeoutS   – seconds of inactivity after which a STOP is sent to
                       de-energise the coils and prevent overheating (default: 2.0);
                       set to 0 to disable
  homeSpeedSteps     – speed in steps/s used during homing (default: 100)
  homeTimeS          – time in seconds the stage moves toward the home position
                       before stopping (default: 5.0)
  homeMonitorCurrent – if True, attempt to detect stall by reading CURRENT during
                       homing (default: False; requires firmware support)

Example setup JSON snippet
---------------------------
  "StepperXY": {
    "managerName": "StepperXYStageManager",
    "managerProperties": {
      "rs232device": "StepperSerial",
      "stepsizeX": 0.5,
      "stepsizeY": 0.5,
      "initialSpeedX": 600,
      "initialSpeedY": 600,
      "swapXY": false,
      "backlashX": 0,
      "backlashY": 0,
      "idleStopTimeoutS": 2.0,
      "homeSpeedSteps": 100,
      "homeTimeS": 5.0
    },
    "axes": ["X", "Y"],
    "forPositioning": true,
    "forScanning":    false,
    "resetOnClose":   false
  }

  "StepperSerial": {
    "managerName":       "SerialManager",
    "managerProperties": {
      "port":              "/dev/cu.usbmodem987ACBFC1",
      "baudrate":          9600,
      "send_termination":  "lf",
      "recv_termination":  "lf",
      "timeout":           1.0
    }
  }
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

from imswitch.imcommon.model import initLogger
from .PositionerManager import PositionerManager

# Default reply timeout when waiting for OK / ERR (seconds)
_REPLY_TIMEOUT = 3.0
# Default speed used when no explicit speed is passed
_DEFAULT_SPEED = 800
# Current threshold (arbitrary units) above which a stall is assumed during homing
_STALL_CURRENT_THRESHOLD = 1000



class StepperXYStageManager(PositionerManager):
    """
    2-axis XY stepper stage controlled via a text-based serial protocol.

    Only X and Y axes are supported.  Position is tracked in software
    (the device only supports relative moves).
    """

    def __init__(self, positionerInfo, name: str, **lowLevelManagers):
        # Initialise base class with zero position for X and Y
        try:
            super().__init__(
                positionerInfo,
                name,
                initialPosition={"X": 0.0, "Y": 0.0, "Z": 0.0},
                initialSpeed={"X": 0.0, "Y": 0.0, "Z": 0.0},
            )
        except Exception as exc:
            print(f"Failed to initialize PositionerManager base class: {exc}")
        self.__logger = initLogger(self, instanceName=name)
        self._commChannel = lowLevelManagers['commChannel']
        self._name = name
        self._lock = threading.Lock()

        # Grab the RS232 manager that provides .query() / .write()
        rs232_key = positionerInfo.managerProperties["rs232device"]
        self._serial = lowLevelManagers["rs232sManager"][rs232_key]

        # Calibration: µm per step
        self._stepsizeX: float = positionerInfo.managerProperties.get("stepsizeX", 1.0)
        self._stepsizeY: float = positionerInfo.managerProperties.get("stepsizeY", 1.0)

        # Default speeds (steps/s)
        init_speed_x: int = int(positionerInfo.managerProperties.get("initialSpeedX", _DEFAULT_SPEED))
        init_speed_y: int = int(positionerInfo.managerProperties.get("initialSpeedY", _DEFAULT_SPEED))
        self._speed["X"] = init_speed_x
        self._speed["Y"] = init_speed_y
        #self._set_device_speed(init_speed_x, init_speed_y) # don't do that here, sample will start moving

        # Optional microstep configuration on startup
        microsteps = positionerInfo.managerProperties.get("microsteps", None)
        self._umPerCycle = (
            positionerInfo.managerProperties.get("umPerElectricalCycleX"),
            positionerInfo.managerProperties.get("umPerElectricalCycleY"),
        )
        power = (
            positionerInfo.managerProperties.get("powerX"),
            positionerInfo.managerProperties.get("powerY"),
        )
        # Firmware without BUSY/POS falls back to time-based waiting
        self._has_busy: bool = True
        props = positionerInfo.managerProperties
        self._approach = {"X": int(props.get("approachDirectionX", 0)),
                          "Y": int(props.get("approachDirectionY", 0))}
        self._approach_overshoot: int = int(props.get("approachOvershootSteps", 8))
        self._settle_s: float = float(props.get("settleMs", 0)) / 1000
        self._max_speed: Optional[int] = (int(props["maxSpeedSteps"])
                                          if props.get("maxSpeedSteps") else None)
        firmware_settings: dict = dict(props.get("firmware", {}) or {})

        # --- X/Y axis swap (for stages mounted 90° rotated) ---
        self._swapXY: bool = bool(positionerInfo.managerProperties.get("swapXY", False))

        # --- Backlash compensation (steps per axis) ---
        self._backlashX: int = int(positionerInfo.managerProperties.get("backlashX", 0))
        self._backlashY: int = int(positionerInfo.managerProperties.get("backlashY", 0))
        # Track last direction to detect reversals (0 = unknown, +1 / -1)
        self._backlash_dir: Dict[str, int] = {"X": 0, "Y": 0}

        # --- Homing parameters ---
        self._home_speed: int = int(positionerInfo.managerProperties.get("homeSpeedSteps", 100))
        self._home_time_s: float = float(positionerInfo.managerProperties.get("homeTimeS", 5.0))
        self._home_monitor_current: bool = bool(
            positionerInfo.managerProperties.get("homeMonitorCurrent", False)
        )

        # --- Anti-overheat idle-stop watchdog ---
        # After this many seconds of inactivity a STOP is sent to de-energise
        # the motor coils.  Set to 0 to disable.
        self._idle_stop_timeout_s: float = float(
            positionerInfo.managerProperties.get("idleStopTimeoutS", 2.0)
        )
        self._last_active_time: float = time.time()
        # Set True while moveForever is active so the watchdog does not interrupt
        self._forever_moving: bool = False
        self._watchdog_stop_evt = threading.Event()
        if self._idle_stop_timeout_s > 0:
            self._watchdog_thread = threading.Thread(
                target=self._idle_watchdog, daemon=True, name=f"{name}-watchdog"
            )
            self._watchdog_thread.start()
            self.__logger.info(
                f"{name}: idle-stop watchdog started (timeout={self._idle_stop_timeout_s}s)"
            )

        # Verify connectivity
        '''
        try:
            self.ping()
            self.__logger.info(f"{name}: device responded to PING")
        except Exception as exc:
            self.__logger.warning(f"{name}: PING failed ({exc}) – continuing anyway")
        '''
        # Apply microstep setting if configured
        if microsteps is not None:
            try:
                self.set_microsteps(int(microsteps))
                self.__logger.info(f"{name}: microsteps set to {microsteps}")
            except Exception as exc:
                self.__logger.warning(f"{name}: could not set microsteps: {exc}")

        # Derive µm/step from the microstep mode actually running on the device
        if any(v is not None for v in self._umPerCycle):
            try:
                n = self.get_microsteps()
                if self._umPerCycle[0] is not None:
                    self._stepsizeX = float(self._umPerCycle[0]) / n
                if self._umPerCycle[1] is not None:
                    self._stepsizeY = float(self._umPerCycle[1]) / n
                self.__logger.info(
                    f"{name}: {n} microsteps/cycle -> "
                    f"{self._stepsizeX:.3f} / {self._stepsizeY:.3f} µm/step (X/Y)"
                )
            except Exception as exc:
                self.__logger.warning(
                    f"{name}: MICROSTEP query failed ({exc}); keeping stepsizeX/Y"
                )

        if power[0] is not None and power[1] is not None:
            try:
                self.set_power(int(power[0]), int(power[1]))
            except Exception as exc:
                self.__logger.warning(f"{name}: could not set power: {exc}")

        # Runtime firmware settings (not persisted in firmware, so re-sent here)
        for key, value in firmware_settings.items():
            try:
                if key.upper().startswith("PHASE_"):
                    axis = key.split("_", 1)[1].upper()
                    for k, v in enumerate(value):
                        self._send(f"PHASE {axis} {k} {int(v)}", log=False)
                else:
                    reply = self._send(f"{key.upper()} {value}")
                    if not reply.startswith("OK"):
                        self.__logger.warning(f"{name}: firmware setting {key}={value}: {reply!r}")
            except Exception as exc:
                self.__logger.warning(f"{name}: firmware setting {key} failed: {exc}")


    # ------------------------------------------------------------------
    # Anti-overheat watchdog
    # ------------------------------------------------------------------

    def _idle_watchdog(self) -> None:
        """
        Background daemon thread.  Sends STOP whenever the stage has been
        idle for more than ``_idle_stop_timeout_s`` seconds, which
        de-energises the motor coils and prevents overheating.
        """
        while not self._watchdog_stop_evt.is_set():
            self._watchdog_stop_evt.wait(timeout=0.5)
            if self._watchdog_stop_evt.is_set():
                break
            if self._forever_moving:
                # moveForever is explicitly running – do not interrupt
                continue
            elapsed = time.time() - self._last_active_time
            if elapsed < self._idle_stop_timeout_s:
                continue
            try:
                # A long non-blocking MOVE or SNAKE must not be cut short.
                if self.is_busy():
                    self._last_active_time = time.time()
                    continue
                # query (not write) so the "OK" reply is consumed; a bare write
                # left it in the buffer and shifted every later reply by a line.
                self._send("STOP", log=False)
                self.__logger.debug("Watchdog: idle STOP sent")
            except Exception as exc:
                self.__logger.warning(f"Watchdog STOP failed: {exc}")
            # Reset timer so we do not spam STOP every 0.5 s
            self._last_active_time = time.time()

    # ------------------------------------------------------------------
    # Low-level serial helpers
    # ------------------------------------------------------------------

    def _send(self, cmd: str, log: bool = True) -> str:
        """
        Send a command string and return the reply.
        Logs a warning if the device responds with ERR.
        Thread-safe.  ``log=False`` keeps status polling out of the log.
        """
        with self._lock:
            if log:
                self.__logger.info(f"TX: {cmd!r}")
            reply = self._serial.query(cmd).strip()
            if log:
                self.__logger.info(f"RX: {reply!r}")
        # Refresh activity timestamp so the watchdog does not interrupt
        self._last_active_time = time.time()
        if reply.startswith("ERR"):
            self.__logger.warning(f"Device error for '{cmd}': {reply}")
        return reply

    def is_busy(self) -> bool:
        """True while the firmware is executing a MOVE/SNAKE or still stepping."""
        if not self._has_busy:
            return False
        reply = self._send("BUSY", log=False)
        if reply not in ("0", "1"):
            self._has_busy = False
            self.__logger.warning(
                f"BUSY not supported by firmware ({reply!r}); "
                "falling back to time-based waiting"
            )
            return False
        return reply == "1"

    def wait_idle(self, timeout_s: float) -> bool:
        """Poll BUSY until the stage stops.  Returns False on timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.is_busy():
                return True
            time.sleep(0.01)
        self.__logger.warning(f"Stage still busy after {timeout_s:.1f}s")
        return False

    def _send_move(self, dx: int, dy: int, speed: Optional[float],
                   wait: bool = True) -> None:
        """Send one MOVE; optionally block until the firmware reports idle.

        A new MOVE replaces a running one in the firmware, so we always wait
        for the previous move to finish first; otherwise the software position
        and the real position diverge.
        """
        if isinstance(speed, tuple):
            speed = speed[0]
        if speed is None:
            speed = max(self._speed["X"], self._speed["Y"]) or _DEFAULT_SPEED
        speed = max(int(speed), 1)  # firmware clamps to its own maximum
        if self._max_speed:
            speed = min(speed, self._max_speed)

        # Firmware ramps at ~1500 steps/s² by default; 2x margin + 2 s.
        duration = max(abs(dx), abs(dy)) / min(speed, 2000) + speed / 1500
        timeout = 2 * duration + 2.0

        self.wait_idle(timeout)
        self._send(f"MOVE {dx} {dy} {speed}")
        if not wait:
            return
        if self._has_busy:
            self.wait_idle(timeout)
        else:
            time.sleep(duration)
        if self._settle_s > 0:
            time.sleep(self._settle_s)

    def _set_device_speed(self, speed_x: int, speed_y: int) -> None:
        """Send V command to set axis speeds (steps/s)."""
        self._send(f"V {speed_x} {speed_y}")

    # ------------------------------------------------------------------
    # PositionerManager interface
    # ------------------------------------------------------------------

    def move(self,
             value: float = 0.0,
             axis: str = "X",
             is_absolute: bool = False,
             is_blocking: bool = True,
             speed: Optional[float] = None,
             **kwargs) -> None:
        """
        Move the stage.

        Parameters
        ----------
        value :
            Distance in µm (converted to steps internally).
        axis :
            "X", "Y", or "XY".  For "XY" value must be a (dx, dy) tuple
            in µm.
        is_absolute :
            If True, move to absolute position (tracked in software).
        speed :
            Speed override in steps/s.  Uses per-axis default if None.
        is_blocking :
            If True, return only after the firmware reports the move done.
        """
        if is_absolute:
            # Convert absolute target to relative steps from current position
            if axis == "X":
                delta_um = value - self._position["X"]
                self._move_xy_steps(delta_um / self._stepsizeX, 0, speed=speed, wait=is_blocking)
                self._position["X"] = value
            elif axis == "Y":
                delta_um = value - self._position["Y"]
                self._move_xy_steps(0, delta_um / self._stepsizeY, speed=speed, wait=is_blocking)
                self._position["Y"] = value
            elif axis == "XY":
                dx_um = value[0] - self._position["X"]
                dy_um = value[1] - self._position["Y"]
                self._move_xy_steps(dx_um / self._stepsizeX,
                                    dy_um / self._stepsizeY,
                                    speed=speed, wait=is_blocking)
                self._position["X"] = value[0]
                self._position["Y"] = value[1]
        else:
            # Relative move in µm
            if axis == "X":
                self._move_xy_steps(value / self._stepsizeX, 0, speed=speed, wait=is_blocking)
                self._position["X"] += value
            elif axis == "Y":
                self._move_xy_steps(0, value / self._stepsizeY, speed=speed, wait=is_blocking)
                self._position["Y"] += value
            elif axis == "XY":
                self._move_xy_steps(value[0] / self._stepsizeX,
                                    value[1] / self._stepsizeY,
                                    speed=speed, wait=is_blocking)
                self._position["X"] += value[0]
                self._position["Y"] += value[1]
            else:
                self.__logger.warning(f"Axis '{axis}' is not supported (only X, Y, XY)")
        # Emit position update in the same format as ESP32StageManager:
        # { positionerName: { "X": ..., "Y": ..., "Z": ... } }
        posDict = {self._name: dict(self._position)}
        self._commChannel.sigUpdateMotorPosition.emit(posDict)


    def _move_xy_steps(self,
                       dx_steps: float,
                       dy_steps: float,
                       speed: Optional[float] = None,
                       wait: bool = True) -> None:
        """
        Send a MOVE command with optional speed, applying X/Y swap and
        backlash compensation.

        Parameters
        ----------
        dx_steps, dy_steps :
            Logical relative displacement in steps (may be float, rounded
            to int).  "Logical" means in the coordinate system that callers
            (and the user) see; X/Y swap is applied here before hitting the
            wire.
        speed :
            Speed in steps/s (uses current default if None).
        """
        # Apply X/Y axis swap (for stages mounted 90° rotated)
        if self._swapXY:
            dx_dev = int(round(dy_steps))
            dy_dev = int(round(dx_steps))
        else:
            dx_dev = int(round(dx_steps))
            dy_dev = int(round(dy_steps))

        if dx_dev == 0 and dy_dev == 0:
            return

        if any(self._approach.values()):
            # Unidirectional approach: a move against the approach direction
            # overshoots the target, then every axis finishes travelling in
            # its approach direction (the friction dead band is taken up
            # before the target, not at it).
            dev_dir = ({"X": self._approach["Y"], "Y": self._approach["X"]} if self._swapXY
                       else self._approach)
            over = {}
            for axis, d in (("X", dx_dev), ("Y", dy_dev)):
                a = dev_dir[axis]
                over[axis] = -a * self._approach_overshoot if (a and d and (d > 0) != (a > 0)) else 0
            if over["X"] or over["Y"]:
                self._send_move(dx_dev + over["X"], dy_dev + over["Y"], speed, wait=True)
                self._send_move(-over["X"], -over["Y"], speed, wait=wait)
            else:
                self._send_move(dx_dev, dy_dev, speed, wait=wait)
            return

        # Backlash compensation: when reversing direction, overshoot by the
        # configured backlash amount then return to the exact target so the
        # final approach is always from the same mechanical side.
        extra_x = 0
        extra_y = 0
        if dx_dev != 0 and (self._backlashX) != 0:
            new_dir = 1 if dx_dev > 0 else -1
            if self._backlash_dir["X"] != 0 and new_dir != self._backlash_dir["X"]:
                # Overshoot in the direction of movement by abs(backlash) steps
                extra_x = (self._backlashX) * new_dir
            self._backlash_dir["X"] = new_dir
        if dy_dev != 0 and (self._backlashY) != 0:
            new_dir = 1 if dy_dev > 0 else -1
            if self._backlash_dir["Y"] != 0 and new_dir != self._backlash_dir["Y"]:
                # Overshoot in the direction of movement by abs(backlash) steps
                extra_y = (self._backlashY) * new_dir
            self._backlash_dir["Y"] = new_dir

        if extra_x != 0 or extra_y != 0:
            # Move to the overshoot position (must finish, or the second
            # MOVE replaces it in the firmware) …
            self._send_move(dx_dev + extra_x, dy_dev + extra_y, speed, wait=True)
            # … then back to the exact logical target
            self._send_move(-extra_x, -extra_y, speed, wait=wait)
        else:
            self._send_move(dx_dev, dy_dev, speed, wait=wait)

    def moveForever(self, speed=(0, 0), is_stop: bool = False) -> None:
        """
        Move the stage indefinitely at the given speed.

        Parameters
        ----------
        speed :
            (speed_x, speed_y) in steps/s.  Set to (0, 0) or pass
            is_stop=True to stop.
        is_stop :
            If True, sends STOP regardless of speed.
        """
        if is_stop or (speed[0] == 0 and speed[1] == 0):
            self.stopAll()
        else:
            sx = int(speed[0])
            sy = int(speed[1])
            # Apply X/Y swap for continuous motion as well
            dev_sx, dev_sy = (sy, sx) if self._swapXY else (sx, sy)
            self._forever_moving = True
            self._set_device_speed(abs(dev_sx), abs(dev_sy))
            # MOVE with large step count to run until STOP is received
            self._send(
                f"MOVE {dev_sx * 9999} {dev_sy * 9999} "
                f"{max(abs(dev_sx), abs(dev_sy), 1)}"
            )

    def setSpeed(self, speed: float, axis: Optional[str] = None) -> None:
        """Update the default speed (steps/s) for the given axis."""
        if axis is None:
            self._speed["X"] = float(speed)
            self._speed["Y"] = float(speed)
            self._set_device_speed(int(speed), int(speed))
        elif axis == "X":
            self._speed["X"] = float(speed)
            self._set_device_speed(int(self._speed["X"]), int(self._speed["Y"]))
        elif axis == "Y":
            self._speed["Y"] = float(speed)
            self._set_device_speed(int(self._speed["X"]), int(self._speed["Y"]))

    def setPosition(self, value: float, axis: str) -> None:
        """Override the tracked software position without moving."""
        if axis in ("X", "Y"):
            self._position[axis] = float(value)

    def getPosition(self) -> Dict[str, float]:
        """Return the current software-tracked position dict."""
        return dict(self._position)

    def stopAll(self) -> None:
        """Send STOP command to immediately halt all motion."""
        self._forever_moving = False
        try:
            self._send("STOP")
        except Exception as exc:
            self.__logger.warning(f"STOP command failed: {exc}")

    def finalize(self) -> None:
        """Stop the stage and tear down the watchdog on shutdown."""
        self._watchdog_stop_evt.set()
        self.stopAll()

    # ------------------------------------------------------------------
    # Device-specific extras
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Send PING and return True if the device replies OK."""
        reply = self._send("PING")
        return "OK" in reply

    def doHome(self, axis, isBlocking=False, homeDirection=None, homeSpeed=None, homeEndstoppolarity=None, homeEndposRelease=None, homeTimeout=None):
        """Home one axis (ExperimentController.homeAllAxes calls X, Y, Z in turn)."""
        if axis not in ("X", "Y"):
            return
        self.home(
            home_speed=homeSpeed,
            home_time_s=homeTimeout,
            monitor_current=self._home_monitor_current,
            axes=(axis,),
        )

    def home(self,
             home_speed: Optional[int] = None,
             home_time_s: Optional[float] = None,
             monitor_current: Optional[bool] = None,
             axes=("X", "Y")) -> None:
        """
        Move the stage toward its home position (-X / -Y end of travel).

        Because there are no limit switches the stage is driven in the
        negative X and Y direction for a fixed time and then stopped.
        The software position is reset to (0, 0) afterwards.

        Parameters
        ----------
        home_speed :
            Speed in steps/s during homing.  Defaults to the
            ``homeSpeedSteps`` config value (default 100).
        home_time_s :
            How long to move before stopping.  Defaults to the
            ``homeTimeS`` config value (default 5.0 s).
        monitor_current :
            If True, poll the device for current draw during the move and
            stop early when a stall is detected (motor hits end-of-travel).
            Defaults to the ``homeMonitorCurrent`` config value.
        axes :
            Which logical axes to home; the others keep their position.
        """
        speed = int(home_speed) if home_speed is not None else self._home_speed
        duration = float(home_time_s) if home_time_s is not None else self._home_time_s
        use_current = (
            monitor_current if monitor_current is not None
            else self._home_monitor_current
        )
        self.__logger.info(
            f"Homing: speed={speed} steps/s, time={duration}s, "
            f"monitorCurrent={use_current}"
        )

        # Disable idle-stop watchdog for the duration so it does not
        # interfere with the deliberately long continuous move.
        self._forever_moving = True
        try:
            #self._set_device_speed(speed, speed)
            # Apply X/Y swap so the physical axes are correct
            # Negative = against the scan direction (approachDirection +1), so
            # the sled is pushed into its -X/-Y stop and the scan then only
            # travels in +X/+Y from there.
            big = speed * 9999  # large enough step count to run the full time
            dev_dx = -big if "X" in axes else 0
            dev_dy = -big if "Y" in axes else 0
            if self._swapXY:
                dev_dx, dev_dy = dev_dy, dev_dx
            self._send(f"MOVE {dev_dx} {dev_dy} {speed}")

            deadline = time.time() + duration
            stalled_x = False
            stalled_y = False
            while time.time() < deadline:
                if use_current:
                    # If the firmware supports a CURRENT command, try to read it.
                    # On stall the current spikes; we stop that axis.
                    try:
                        current_reply = self._send("CURRENT")
                        # Expected format: "<ix> <iy>" or similar
                        tokens = current_reply.split()
                        if len(tokens) >= 2:
                            ix = float(tokens[0])
                            iy = float(tokens[1])
                            # Heuristic: if current exceeds 1.5× idle level, stall
                            if ix > _STALL_CURRENT_THRESHOLD:
                                stalled_x = True
                            if iy > _STALL_CURRENT_THRESHOLD:
                                stalled_y = True
                        if stalled_x and stalled_y:
                            self.__logger.info("Homing: both axes stalled – stopping early")
                            break
                    except Exception:
                        pass  # Firmware may not support CURRENT; ignore
                time.sleep(0.1)

            self.stopAll()
        finally:
            self._forever_moving = False

        # Reset software position and backlash tracking
        for axis in axes:
            self._position[axis] = 0.0
            self._backlash_dir[axis] = 0
        self.__logger.info(f"Homing complete – {'/'.join(axes)} reset to 0")
        self._commChannel.sigUpdateMotorPosition.emit({self._name: dict(self._position)})

    def get_microsteps(self) -> int:
        """Query the current microstep divisor from the device."""
        reply = self._send("MICROSTEP")
        # Device returns the numeric value, e.g. "32" or "32\nOK"
        for token in reply.split():
            try:
                return int(token)
            except ValueError:
                continue
        raise ValueError(f"Could not parse MICROSTEP reply: {reply!r}")

    def getScanHints(self) -> dict:
        """What a tile scan should know about this stage (see PCB-stage bench E3/E6):
        tile steps that are whole full steps (cycle / 4)
        are the most repeatable, and one approach direction avoids the
        reversal dead band."""
        n = None
        try:
            n = self.get_microsteps()
        except Exception:
            pass
        per_full = (n // 4) if n else None
        return {
            "stepsizeUm": {"X": self._stepsizeX, "Y": self._stepsizeY},
            "microsteps": n,
            "fullStepUm": ({"X": self._stepsizeX * per_full, "Y": self._stepsizeY * per_full}
                           if per_full else None),
            "approachDirection": dict(self._approach),
            "approachOvershootSteps": self._approach_overshoot,
            "settleMs": self._settle_s * 1000,
            "maxSpeedSteps": self._max_speed,
            "recommendedMinOverlap": 0.25,
            "recommendedPattern": "raster" if any(self._approach.values()) else "snake",
        }

    def get_position_steps(self) -> Tuple[int, int]:
        """Firmware step counts (device axes, i.e. before swapXY)."""
        reply = self._send("POS")
        x, y = reply.split()
        return int(x), int(y)

    def set_power(self, x: int, y: int) -> None:
        """Per-axis peak duty 0..32767 (0 = firmware default OBJECT3002)."""
        self._send(f"POWER {int(x)} {int(y)}")

    def get_power(self) -> Tuple[int, int]:
        x, y = self._send("POWER").split()
        return int(x), int(y)

    # ---------------------------------------------------------------- calibration
    def move_steps(self, dx: int, dy: int, speed: Optional[float] = None) -> None:
        """Blocking raw move in logical µsteps: no approach overshoot, no
        backlash compensation (for calibration, which measures exactly those)."""
        dev = (int(dy), int(dx)) if self._swapXY else (int(dx), int(dy))
        if dev != (0, 0):
            self._send_move(dev[0], dev[1], speed, wait=True)
        self._position["X"] += dx * self._stepsizeX
        self._position["Y"] += dy * self._stepsizeY
        self._commChannel.sigUpdateMotorPosition.emit({self._name: dict(self._position)})

    def set_axis_power(self, x: int, y: int) -> None:
        """Per-axis drive for the logical X/Y axes (follows swapXY)."""
        self.set_power(*((y, x) if self._swapXY else (x, y)))

    def get_axis_power(self) -> Tuple[int, int]:
        dx, dy = self.get_power()
        return (dy, dx) if self._swapXY else (dx, dy)

    def apply_calibration(self, powerX: Optional[int] = None, powerY: Optional[int] = None,
                          umPerStepX: Optional[float] = None, umPerStepY: Optional[float] = None,
                          approachOvershootSteps: Optional[int] = None) -> dict:
        """Apply calibrated values now; returns the managerProperties that
        reproduce them at the next start (for saving to the setup file)."""
        props = {}
        px, py = self.get_axis_power()
        if powerX or powerY:
            px, py = int(powerX or px), int(powerY or py)
            self.set_axis_power(px, py)
            props.update(powerX=px, powerY=py)
        n = None
        for axis, ups in (("X", umPerStepX), ("Y", umPerStepY)):
            if ups:
                setattr(self, f"_stepsize{axis}", float(ups))
                props[f"stepsize{axis}"] = round(float(ups), 4)
                n = n or self.get_microsteps()
                props[f"umPerElectricalCycle{axis}"] = round(float(ups) * n, 1)
        if approachOvershootSteps:
            self._approach_overshoot = int(approachOvershootSteps)
            props["approachOvershootSteps"] = int(approachOvershootSteps)
        return props

    def set_microsteps(self, n: int) -> None:
        """Store the microstep divisor; only takes effect after a device reboot."""
        self._send(f"MICROSTEP {n}")

    def snake_scan(self,
                   nx: int, ny: int,
                   steps_x: int, steps_y: int,
                   speed: int,
                   pause_ms: int,
                   hold_pct: Optional[int] = None) -> None:
        """
        Execute a hardware snake scan on the device.

        Parameters
        ----------
        nx, ny       : number of columns / rows
        steps_x, steps_y : step size per move (steps)
        speed        : movement speed (steps/s)
        pause_ms     : pause between moves (ms)
        hold_pct     : optional hold-torque percentage during pauses (0–100)
        """
        if hold_pct is not None:
            cmd = f"SNAKE {nx} {ny} {steps_x} {steps_y} {speed} {pause_ms} {hold_pct}"
        else:
            cmd = f"SNAKE {nx} {ny} {steps_x} {steps_y} {speed} {pause_ms}"
        self._send(cmd)

        # Update software position after snake completes
        # (net displacement depends on nx parity — even rows end at start column)
        total_y_steps = (ny - 1) * steps_y
        self._position["X"] += 0  # net X is 0 for even number of rows
        if nx % 2 == 1:
            self._position["X"] += (nx - 1) * steps_x * self._stepsizeX
        self._position["Y"] += total_y_steps * self._stepsizeY

    def help(self) -> str:
        """Return the device help text."""
        with self._lock:
            reply = self._serial.query("HELP")
            # HELP is multi-line; drain the rest so later replies stay in sync
            reply += self._serial.read(4096, timeout=0.3).decode(errors="replace")
        return reply
