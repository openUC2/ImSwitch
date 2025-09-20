"""
Mock FocusLockController for testing and systems without focus lock hardware.
Provides the same API as FocusLockController but returns default/safe values.
"""

import time
from typing import Dict, Any, Optional, List
from imswitch.imcommon.model import initLogger, APIExport
from imswitch.imcommon.framework import Signal
from ..basecontrollers import ImConWidgetController


class MockFocusLockController(ImConWidgetController):
    """
    Mock implementation of FocusLockController for testing and systems without hardware.
    Always returns settled=True and zero offsets to allow testing of the focus map system.
    """
    
    # Signals for compatibility with real FocusLockController
    sigFocusLockStateChanged = Signal(dict)
    sigFocusValueChanged = Signal(float)
    sigFocusLockToggled = Signal(bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        
        # Mock state
        self._locked = False
        self._measuring = False
        self._settled = True
        self._focus_value = 100.0  # Mock focus metric value
        self._z_position = 0.0
        self._error_um = 0.0
        self._setpoint = 100.0
        
        # Mock parameters
        self._settle_band_um = 1.0
        self._settle_timeout_ms = 1500
        self._settle_window_ms = 200
        
        self._logger.info("MockFocusLockController initialized")
    
    @APIExport(runOnUIThread=True)
    def enableFocusLock(self, enable: bool = True) -> bool:
        """Mock enable/disable focus lock."""
        self._locked = enable
        self._logger.info(f"Mock focus lock {'enabled' if enable else 'disabled'}")
        self.sigFocusLockStateChanged.emit(self.getFocusLockState())
        return True
    
    @APIExport(runOnUIThread=True)
    def isFocusLocked(self) -> bool:
        """Mock focus lock status."""
        return self._locked
    
    @APIExport(runOnUIThread=True)
    def getFocusLockState(self) -> Dict[str, Any]:
        """Mock focus lock state."""
        return {
            "locked": self._locked,
            "measuring": self._measuring,
            "settled": self._settled,
            "focus_value": self._focus_value,
            "z_position": self._z_position,
            "error_um": self._error_um,
            "setpoint": self._setpoint
        }
    
    @APIExport(runOnUIThread=True)
    def startFocusMeasurement(self) -> bool:
        """Mock start focus measurement."""
        self._measuring = True
        self._logger.info("Mock focus measurement started")
        return True
    
    @APIExport(runOnUIThread=True)
    def stopFocusMeasurement(self) -> bool:
        """Mock stop focus measurement."""
        self._measuring = False
        self._logger.info("Mock focus measurement stopped")
        return True
    
    @APIExport(runOnUIThread=True)
    def getCurrentFocusValue(self) -> float:
        """Mock current focus value."""
        return self._focus_value
    
    @APIExport(runOnUIThread=True)
    def isSettled(self) -> bool:
        """Mock settled state - always returns True for testing."""
        return self._settled
    
    @APIExport(runOnUIThread=True)
    def getFocusError(self) -> float:
        """Mock focus error - always returns 0 for testing."""
        return self._error_um
    
    @APIExport(runOnUIThread=True)
    def getSettleParams(self) -> Dict[str, Any]:
        """Mock settle parameters."""
        return {
            "settle_band_um": self._settle_band_um,
            "settle_timeout_ms": self._settle_timeout_ms,
            "settle_window_ms": self._settle_window_ms
        }
    
    @APIExport(runOnUIThread=True)
    def setSettleParams(self, settle_band_um: Optional[float] = None,
                       settle_timeout_ms: Optional[int] = None,
                       settle_window_ms: Optional[int] = None) -> Dict[str, Any]:
        """Mock set settle parameters."""
        if settle_band_um is not None:
            self._settle_band_um = float(settle_band_um)
        if settle_timeout_ms is not None:
            self._settle_timeout_ms = int(settle_timeout_ms)
        if settle_window_ms is not None:
            self._settle_window_ms = int(settle_window_ms)
        
        self._logger.info(f"Mock settle params updated: band={self._settle_band_um}um, "
                         f"timeout={self._settle_timeout_ms}ms, window={self._settle_window_ms}ms")
        
        return self.getSettleParams()
    
    @APIExport(runOnUIThread=True)
    def performAutofocus(self, z_min: Optional[float] = None, 
                        z_max: Optional[float] = None,
                        z_step: Optional[float] = None) -> Dict[str, Any]:
        """
        Mock autofocus operation.
        
        Args:
            z_min: Minimum Z position for autofocus range
            z_max: Maximum Z position for autofocus range  
            z_step: Step size for autofocus scan
            
        Returns:
            Dictionary with autofocus results
        """
        # Simulate autofocus delay
        time.sleep(0.1)
        
        # Mock successful autofocus result
        mock_result = {
            "success": True,
            "z_position": self._z_position,
            "focus_value": self._focus_value,
            "iterations": 5,
            "time_ms": 100
        }
        
        self._logger.info(f"Mock autofocus completed: {mock_result}")
        return mock_result
    
    @APIExport(runOnUIThread=True)
    def lockFocus(self, z_position: Optional[float] = None) -> bool:
        """Mock lock focus at current or specified position."""
        if z_position is not None:
            self._z_position = float(z_position)
        
        self._locked = True
        self._settled = True
        self._setpoint = self._focus_value
        self._error_um = 0.0
        
        self._logger.info(f"Mock focus locked at Z={self._z_position}um")
        self.sigFocusLockStateChanged.emit(self.getFocusLockState())
        return True
    
    @APIExport(runOnUIThread=True)
    def unlockFocus(self) -> bool:
        """Mock unlock focus."""
        self._locked = False
        self._settled = False
        
        self._logger.info("Mock focus unlocked")
        self.sigFocusLockStateChanged.emit(self.getFocusLockState())
        return True
    
    @APIExport(runOnUIThread=True)
    def waitForSettle(self, timeout_ms: Optional[int] = None) -> bool:
        """
        Mock wait for focus to settle.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            True if settled within timeout, False otherwise
        """
        if timeout_ms is None:
            timeout_ms = self._settle_timeout_ms
        
        # Mock immediate settle for testing
        self._settled = True
        self._logger.info(f"Mock focus settled immediately (timeout was {timeout_ms}ms)")
        return True
    
    @APIExport(runOnUIThread=True)
    def getZPosition(self) -> float:
        """Mock get current Z position."""
        return self._z_position
    
    @APIExport(runOnUIThread=True)
    def setZPosition(self, z_position: float) -> bool:
        """Mock set Z position."""
        self._z_position = float(z_position)
        self._logger.info(f"Mock Z position set to {self._z_position}um")
        return True
    
    # Legacy compatibility methods
    @APIExport(runOnUIThread=True)
    def toggleFocus(self, toLock: Optional[bool] = None) -> bool:
        """Mock toggle focus lock."""
        if toLock is None:
            toLock = not self._locked
        return self.enableFocusLock(toLock)
    
    def __del__(self):
        """Cleanup mock controller."""
        try:
            self._logger.info("MockFocusLockController cleanup")
        except:
            pass