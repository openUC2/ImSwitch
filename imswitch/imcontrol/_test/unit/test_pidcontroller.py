"""
Unit tests for PID Controller module.
"""

import pytest
import numpy as np
from imswitch.imcontrol.controller.pidcontroller import PIDController


class TestPIDController:
    """Test the extracted PID controller."""
    
    def test_pid_initialization(self):
        """Test PID controller initialization."""
        pid = PIDController(
            set_point=10.0,
            kp=1.0,
            ki=0.5,
            kd=0.1,
            sample_time=0.1
        )
        
        assert pid.set_point == 10.0
        assert pid.kp == 1.0
        assert pid.ki == 0.5
        assert pid.kd == 0.1
        assert pid.dt == 0.1
        assert pid.integral == 0.0
        assert pid.last_measurement is None

    def test_pid_proportional_only(self):
        """Test PID controller with only proportional gain."""
        pid = PIDController(set_point=10.0, kp=2.0, ki=0.0, kd=0.0)
        
        # Test with measurement below setpoint
        output = pid.update(8.0)
        expected = 2.0 * (10.0 - 8.0)  # kp * error
        assert output == expected
        
        # Test with measurement above setpoint
        output = pid.update(12.0)
        expected = 2.0 * (10.0 - 12.0)  # kp * error
        assert output == expected

    def test_pid_integral_accumulation(self):
        """Test integral term accumulation."""
        dt = 0.1
        pid = PIDController(set_point=10.0, kp=0.0, ki=1.0, kd=0.0, sample_time=dt)
        
        # First update
        output1 = pid.update(8.0)
        expected_integral = (10.0 - 8.0) * dt  # error * dt
        expected_output = 1.0 * expected_integral  # ki * integral
        assert abs(output1 - expected_output) < 1e-10
        
        # Second update - integral should accumulate
        output2 = pid.update(8.0)
        expected_integral = 2 * (10.0 - 8.0) * dt  # accumulated error
        expected_output = 1.0 * expected_integral
        assert abs(output2 - expected_output) < 1e-10

    def test_pid_derivative_on_measurement(self):
        """Test derivative term calculated on measurement change."""
        dt = 0.1
        pid = PIDController(
            set_point=10.0, 
            kp=0.0, 
            ki=0.0, 
            kd=1.0, 
            sample_time=dt,
            derivative_filter_alpha=0.0  # No filtering for test
        )
        
        # First update - no derivative
        output1 = pid.update(8.0)
        assert output1 == 0.0  # No previous measurement
        
        # Second update - should have derivative component
        output2 = pid.update(9.0)  # measurement increased by 1.0
        derivative = -(9.0 - 8.0) / dt  # negative because it's derivative of measurement
        expected_output = 1.0 * derivative  # kd * derivative
        assert abs(output2 - expected_output) < 1e-10

    def test_pid_anti_windup(self):
        """Test integral anti-windup limiting."""
        dt = 0.1
        integral_limit = 5.0
        pid = PIDController(
            set_point=10.0,
            kp=0.0,
            ki=1.0,
            kd=0.0,
            sample_time=dt,
            integral_limit=integral_limit
        )
        
        # Feed large error to try to wind up integral
        large_error = 100.0
        for _ in range(100):  # Many iterations
            pid.update(10.0 - large_error)
        
        # Integral should be clamped
        assert abs(pid.integral) <= integral_limit

    def test_pid_parameter_updates(self):
        """Test dynamic parameter updates."""
        pid = PIDController(set_point=10.0, kp=1.0, ki=0.5, kd=0.1)
        
        # Test set_parameters
        pid.set_parameters(kp=2.0, ki=1.0)
        assert pid.kp == 2.0
        assert pid.ki == 1.0
        
        # Test update_parameters
        pid.update_parameters(kd=0.2, sample_time=0.05, set_point=15.0)
        assert pid.kd == 0.2
        assert pid.dt == 0.05
        assert pid.set_point == 15.0

    def test_pid_reset(self):
        """Test controller reset functionality."""
        pid = PIDController(set_point=10.0, kp=1.0, ki=0.5, kd=0.1)
        
        # Generate some history
        pid.update(8.0)
        pid.update(9.0)
        
        # Verify state is not initial
        assert pid.integral != 0.0
        assert pid.last_measurement is not None
        
        # Reset and verify
        pid.reset()
        assert pid.integral == 0.0
        assert pid.last_measurement is None
        assert pid.last_output == 0.0
        assert pid.derivative_filtered == 0.0

    def test_pid_output_filtering(self):
        """Test output low-pass filtering."""
        pid = PIDController(
            set_point=10.0,
            kp=1.0,
            ki=0.0,
            kd=0.0,
            output_lowpass_alpha=0.5
        )
        
        # First update
        output1 = pid.update(8.0)
        raw_output = 1.0 * (10.0 - 8.0)  # No filtering on first update
        assert output1 == raw_output
        
        # Second update with same measurement
        output2 = pid.update(8.0)
        # Output should be filtered: 0.5 * previous + 0.5 * raw
        expected = 0.5 * output1 + 0.5 * raw_output
        assert abs(output2 - expected) < 1e-10

    def test_pid_combined_terms(self):
        """Test PID with all terms active."""
        dt = 0.1
        pid = PIDController(
            set_point=10.0,
            kp=1.0,
            ki=0.5,
            kd=0.2,
            sample_time=dt,
            derivative_filter_alpha=0.0  # No filtering for predictable test
        )
        
        # First measurement
        measurement1 = 8.0
        error1 = 10.0 - measurement1
        output1 = pid.update(measurement1)
        
        # Should only have proportional term (no previous measurement for derivative)
        expected1 = 1.0 * error1
        assert abs(output1 - expected1) < 1e-10
        
        # Second measurement
        measurement2 = 8.5
        error2 = 10.0 - measurement2
        output2 = pid.update(measurement2)
        
        # Calculate expected components
        integral = error1 * dt + error2 * dt
        derivative = -(measurement2 - measurement1) / dt  # On measurement
        expected2 = 1.0 * error2 + 0.5 * integral + 0.2 * derivative
        
        assert abs(output2 - expected2) < 1e-10

    def test_pid_get_state(self):
        """Test state retrieval for monitoring."""
        pid = PIDController(set_point=10.0, kp=1.0, ki=0.5, kd=0.1)
        pid.update(8.0)
        
        state = pid.get_state()
        
        assert state["set_point"] == 10.0
        assert state["kp"] == 1.0
        assert state["ki"] == 0.5
        assert state["kd"] == 0.1
        assert state["last_measurement"] == 8.0
        assert "integral" in state
        assert "last_output" in state

    def test_pid_setpoint_update(self):
        """Test setpoint updates without state reset."""
        pid = PIDController(set_point=10.0, kp=1.0, ki=0.5, kd=0.1)
        
        # Generate some state
        pid.update(8.0)
        pid.update(9.0)
        old_integral = pid.integral
        old_measurement = pid.last_measurement
        
        # Update setpoint
        pid.set_setpoint(15.0)
        
        # Setpoint should change but state should remain
        assert pid.set_point == 15.0
        assert pid.integral == old_integral
        assert pid.last_measurement == old_measurement

    def test_pid_edge_cases(self):
        """Test edge cases and robustness."""
        # Test minimum sample time enforcement
        pid = PIDController(set_point=10.0, sample_time=0.0)
        assert pid.dt > 0.0  # Should be enforced to minimum
        
        # Test with zero gains
        pid = PIDController(set_point=10.0, kp=0.0, ki=0.0, kd=0.0)
        output = pid.update(5.0)
        assert output == 0.0
        
        # Test with very large values
        pid = PIDController(set_point=1e6, kp=1.0)
        output = pid.update(0.0)
        assert output == 1e6