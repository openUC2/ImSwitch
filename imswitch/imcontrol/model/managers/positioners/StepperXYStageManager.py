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
  PING                – connectivity check
  HELP                – print help text
  MICROSTEP           – query current microstep setting
  MICROSTEP N         – set microstep to N (stored in flash)
  SNAKE nx ny sx sy speed pause [holdPct]
                      – execute a snake scan:
                        nx/ny    : number of steps in X/Y
                        sx/sy    : step sizes in steps
                        speed    : movement speed (steps/s)
                        pause    : pause between moves (ms)
                        holdPct  : optional hold-torque % during pauses

Configuration keys (managerProperties)
---------------------------------------
  rs232device   – key into rs232sManager (must be a SerialManager/RS232Manager)
  stepsizeX     – µm per step in X (default: 1.0)
  stepsizeY     – µm per step in Y (default: 1.0)
  initialSpeedX – default speed in steps/s for X (default: 800)
  initialSpeedY – default speed in steps/s for Y (default: 800)
  microsteps    – microstep divisor to write on startup (optional)

Example setup JSON snippet
---------------------------
  "StepperXY": {
    "managerName": "StepperXYStageManager",
    "managerProperties": {
      "rs232device": "StepperSerial",
      "stepsizeX": 0.5,
      "stepsizeY": 0.5,
      "initialSpeedX": 600,
      "initialSpeedY": 600
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
                initialPosition={"X": 0.0, "Y": 0.0},
                initialSpeed={"X": 0.0, "Y": 0.0},
            )
        except Exception as exc:
            print(f"Failed to initialize PositionerManager base class: {exc}")
        self.__logger = initLogger(self, instanceName=name)
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

        # Optional microstep configuration on startup
        microsteps = positionerInfo.managerProperties.get("microsteps", None)

        # Verify connectivity
        try:
            self.ping()
            self.__logger.info(f"{name}: device responded to PING")
        except Exception as exc:
            self.__logger.warning(f"{name}: PING failed ({exc}) – continuing anyway")

        # Apply microstep setting if configured
        if microsteps is not None:
            try:
                self.set_microsteps(int(microsteps))
                self.__logger.info(f"{name}: microsteps set to {microsteps}")
            except Exception as exc:
                self.__logger.warning(f"{name}: could not set microsteps: {exc}")


    # ------------------------------------------------------------------
    # Low-level serial helpers
    # ------------------------------------------------------------------

    def _send(self, cmd: str) -> str:
        """
        Send a command string and return the reply.
        Raises RuntimeError if the device responds with ERR.
        Thread-safe.
        """
        with self._lock:
            self.__logger.info(f"TX: {cmd!r}")
            reply = self._serial.query(cmd).strip()
            self.__logger.info(f"RX: {reply!r}")
        if reply.startswith("ERR"):
            print(f"Device error for '{cmd}': {reply}")
        return reply

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
            Passed through; serial commands are inherently synchronous.
        """
        if is_absolute:
            # Convert absolute target to relative steps from current position
            if axis == "X":
                delta_um = value - self._position["X"]
                self._move_xy_steps(delta_um / self._stepsizeX, 0, speed=speed)
                self._position["X"] = value
            elif axis == "Y":
                delta_um = value - self._position["Y"]
                self._move_xy_steps(0, delta_um / self._stepsizeY, speed=speed)
                self._position["Y"] = value
            elif axis == "XY":
                dx_um = value[0] - self._position["X"]
                dy_um = value[1] - self._position["Y"]
                self._move_xy_steps(dx_um / self._stepsizeX,
                                    dy_um / self._stepsizeY,
                                    speed=speed)
                self._position["X"] = value[0]
                self._position["Y"] = value[1]
        else:
            # Relative move in µm
            if axis == "X":
                self._move_xy_steps(value / self._stepsizeX, 0, speed=speed)
                self._position["X"] += value
            elif axis == "Y":
                self._move_xy_steps(0, value / self._stepsizeY, speed=speed)
                self._position["Y"] += value
            elif axis == "XY":
                self._move_xy_steps(value[0] / self._stepsizeX,
                                    value[1] / self._stepsizeY,
                                    speed=speed)
                self._position["X"] += value[0]
                self._position["Y"] += value[1]
            else:
                self.__logger.warning(f"Axis '{axis}' is not supported (only X, Y, XY)")

    def _move_xy_steps(self,
                       dx_steps: float,
                       dy_steps: float,
                       speed: Optional[float] = None) -> None:
        """
        Send a MOVE command with optional speed.

        Parameters
        ----------
        dx_steps, dy_steps :
            Relative displacement in steps (may be float, rounded to int).
        speed :
            Speed in steps/s (uses current default if None).
        """
        dx = int(round(dx_steps))
        dy = int(round(dy_steps))
        if dx == 0 and dy == 0:
            return
        if speed is not None:
            cmd = f"MOVE {dx} {dy} {int(speed)}"
        else:
            cmd = f"MOVE {dx} {dy}"
        self._send(cmd)

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
            self._set_device_speed(sx, sy)
            # MOVE with large step count to run until STOP
            # (device will stop when STOP is received or when steps run out)
            self._send(f"MOVE {sx * 9999} {sy * 9999} {max(abs(sx), abs(sy), 1)}")

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
        try:
            self._send("STOP")
        except Exception as exc:
            self.__logger.warning(f"STOP command failed: {exc}")

    def finalize(self) -> None:
        """Stop the stage on shutdown."""
        self.stopAll()

    # ------------------------------------------------------------------
    # Device-specific extras
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Send PING and return True if the device replies OK."""
        reply = self._send("PING")
        return "OK" in reply

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

    def set_microsteps(self, n: int) -> None:
        """Set the microstep divisor (stored in flash on the device)."""
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
        return reply
