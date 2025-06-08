from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import LiveUpdatedController
from imswitch.imcommon.framework import Signal
import numpy as np
import time

class TriggerAcquisitionController(LiveUpdatedController):
    """
    Simple controller for switching between software- and hardware-trigger
    acquisition on the current detector (camera).
    • Software trigger → emits/returns one frame.
    • Hardware trigger → fills a ring buffer and returns the whole stack.
    """

    sigFrame  = Signal(np.ndarray)   # single software-trigger frame
    sigBuffer = Signal(np.ndarray)   # hardware-trigger stack

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger  = initLogger(self)
        names         = self._master.detectorsManager.getAllDeviceNames()
        if not names:
            raise RuntimeError("No detector found")
        self.detector = self._master.detectorsManager[names[0]]

    # ── trigger-mode helpers ────────────────────────────────────────────────────
    def _stop_live(self):
        try:
            self._commChannel.sigStopLiveAcquisition.emit(True)
        except Exception:
            pass                                            # no live view running

    def _flush(self):
        try:
            self.detector.flushBuffers()
        except Exception:
            pass

    # ── public API ──────────────────────────────────────────────────────────────
    @APIExport()
    def setSoftwareTrigger(self):
        """Put camera in software-trigger (“internal”) mode."""
        self._stop_live()
        self.detector.setParameter("trigger_source", "Internal trigger")
        self.detector.setParameter("buffer_size", -1)       # disable hw buffer
        self._flush()

    @APIExport()
    def setHardwareTrigger(self, trigger_source:str="External trigger"):
        """
        Arm camera for external line-trigger acquisition and pre-allocate the
        on-board buffer.

        """
        self._stop_live()
        self.detector.setParameter("trigger_source", "External trigger")
        self._flush()

    # ── acquisition ─────────────────────────────────────────────────────────────
    @APIExport()
    def snap(self):
        """
        Fire one software trigger and return the newest frame.
        Controller emits `sigFrame` as well.
        """
        self.detector.sendSoftwareTrigger()
        frame = self.detector.getLatestFrame()
        self.sigFrame.emit(frame)
        return frame

    @APIExport()
    def fetchBuffer(self):
        """
        Retrieve the complete hardware-trigger stack from the camera buffer.
        Controller emits `sigBuffer` as well.
        """
        stack = self.detector.getChunk()
        self._flush()
        if stack is None:
            self._logger.warning("No stack received")
            return None
        arr = np.asarray(stack)
        self.sigBuffer.emit(arr)
        return arr
