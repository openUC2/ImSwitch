"""Grating spectrometer on an area camera: pipeline, calibration, wizard, simulator.

Independent of ImSwitch (numpy and the standard library; h5py only to save HDF5), so
the same code runs in a notebook. ``SpectrometerManager`` and
``SpectrometerController`` wire it to a camera, the lasers and the LED matrix.
See docs/SPECTROMETER.md.
"""
from .calibration import (Calibration, GratingDesign, alignment_plan, make_calibration,
                          nm_to_k, nm_to_raman_shift, nm_to_wavenumber, raman_shift_to_nm,
                          resample_linear_k, turn_from_line)
from .pipeline import ProcessingSettings, find_slit_rows, measure_smile, process, stack_mean
from .simulator import SimulationSettings, SpectrometerSimulator, emission_for_sources
from .wizard import CalibrationSource, CalibrationWizard, WizardSettings

__all__ = [
    "Calibration", "GratingDesign", "alignment_plan", "make_calibration", "nm_to_k",
    "nm_to_raman_shift", "nm_to_wavenumber", "raman_shift_to_nm", "resample_linear_k",
    "turn_from_line", "ProcessingSettings", "find_slit_rows", "measure_smile", "process",
    "stack_mean", "SimulationSettings", "SpectrometerSimulator", "emission_for_sources",
    "CalibrationSource", "CalibrationWizard", "WizardSettings",
]
