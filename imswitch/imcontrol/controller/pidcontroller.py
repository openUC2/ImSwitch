"""
PID Controller module for ImSwitch focus lock functionality.

Extracted from FocusLockController for better modularity and testability.
Implements discrete 2-DOF style PID with derivative on measurement and anti-windup.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PIDController:
    """
    Discrete PID controller with advanced features:
    - 2-DOF style (beta on setpoint can be implemented by caller)
    - Derivative on measurement (avoids derivative kick)
    - Anti-windup with integral clamping
    - Output low-pass filtering
    - Configurable sample time
    """

    def __init__(
        self,
        set_point: float,
        kp: float = 1.,
        ki: float = 0.0,
        kd: float = 0.0,
        sample_time: float = 0.1,
        integral_limit: float = 100.0,
        output_lowpass_alpha: float = 0.0,
        derivative_filter_alpha: float = 0.85,
    ):
        """
        Initialize PID controller.
        
        Args:
            set_point: Target value
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            sample_time: Controller sample time in seconds
            integral_limit: Anti-windup limit for integral term
            output_lowpass_alpha: EMA smoothing factor for output (0=no filter, 1=max filter)
            derivative_filter_alpha: EMA smoothing for derivative term
        """
        self.set_point = float(set_point)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = max(float(sample_time), 1e-6)
        self.integral_limit = float(integral_limit)
        self.output_alpha = float(output_lowpass_alpha)
        self.derivative_alpha = float(derivative_filter_alpha)
        
        # Internal state
        self.integral = 0.0
        self.last_measurement: Optional[float] = None
        self.last_output = 0.0
        self.derivative_filtered = 0.0
        
        logger.debug(f"PID initialized: Kp={kp}, Ki={ki}, Kd={kd}, dt={sample_time}")

    def set_parameters(self, kp: float, ki: float) -> None:
        """Set proportional and integral gains."""
        self.kp = float(kp)
        self.ki = float(ki)
        logger.debug(f"PID parameters updated: Kp={kp}, Ki={ki}")

    def update_parameters(self, **kwargs) -> None:
        """Update any PID parameters dynamically."""
        for key, value in kwargs.items():
            if key == "kp":
                self.kp = float(value)
            elif key == "ki":
                self.ki = float(value)
            elif key == "kd":
                self.kd = float(value)
            elif key == "sample_time":
                self.dt = max(float(value), 1e-6)
            elif key == "integral_limit":
                self.integral_limit = float(value)
            elif key == "output_lowpass_alpha":
                self.output_alpha = float(value)
            elif key == "set_point":
                self.set_point = float(value)
            elif key == "derivative_filter_alpha":
                self.derivative_alpha = float(value)
            else:
                logger.warning(f"Unknown PID parameter: {key}")

    def update(self, measurement: float) -> float:
        """
        Update PID controller with new measurement.
        
        Args:
            measurement: Current process value
            
        Returns:
            Controller output
        """
        measurement = float(measurement)
        
        # Proportional term
        error = self.set_point - measurement
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        
        # Derivative term (on measurement to avoid derivative kick)
        if self.last_measurement is None:
            derivative_raw = 0.0
        else:
            derivative_raw = (measurement - self.last_measurement) / self.dt
        
        # Apply derivative filtering
        self.derivative_filtered = (
            self.derivative_alpha * self.derivative_filtered +
            (1.0 - self.derivative_alpha) * derivative_raw
        )
        
        self.last_measurement = measurement
        
        # Calculate raw output
        output_raw = (
            self.kp * error +
            self.ki * self.integral -
            self.kd * self.derivative_filtered
        )
        
        # Apply output filtering if enabled
        if self.output_alpha > 0.0:
            self.last_output = (
                self.output_alpha * self.last_output +
                (1.0 - self.output_alpha) * output_raw
            )
        else:
            self.last_output = output_raw
            
        return self.last_output

    def reset(self) -> None:
        """Reset controller internal state."""
        self.integral = 0.0
        self.last_measurement = None
        self.last_output = 0.0
        self.derivative_filtered = 0.0
        logger.debug("PID controller reset")

    def get_state(self) -> dict:
        """Get current controller state for debugging/monitoring."""
        return {
            "set_point": self.set_point,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "dt": self.dt,
            "integral": self.integral,
            "last_measurement": self.last_measurement,
            "last_output": self.last_output,
            "derivative_filtered": self.derivative_filtered,
        }

    def set_setpoint(self, setpoint: float) -> None:
        """Update setpoint without resetting controller state."""
        self.set_point = float(setpoint)
        logger.debug(f"PID setpoint updated to: {setpoint}")