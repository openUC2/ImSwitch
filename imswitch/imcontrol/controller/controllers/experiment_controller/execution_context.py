"""
Runtime execution context for the wellplate experiment dispatch.

``ExecutionContext`` collects the values that ``startWellplateExperiment``
computes once at the start of a run (snake tiles, Z offsets, resolved
illumination lists, output paths, initial stage position, …) and passes
them as a single object to :class:`ExperimentNormalMode` /
:class:`ExperimentPerformanceMode`, instead of a 30-keyword call site.

The original keyword-argument interface of ``execute_experiment`` is
preserved – callers may still pass kwargs directly – but the ``Experiment``
pydantic model is the source of truth for everything that lives on
``mExperiment.parameterValue``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Experiment


@dataclass
class ExecutionContext:
    """Runtime values needed by ``execute_experiment``.

    Anything that can be derived from ``self.experiment.parameterValue``
    is intentionally **not** duplicated here – the execution-mode classes
    read directly from the model so the contract stays single-source.
    """

    experiment: Experiment

    # Acquisition plan
    snake_tiles: List[List[Dict[str, Any]]]
    z_positions: List[float]                  # relative offsets (µm) from each tile's base Z

    # Resolved illumination/exposure/gain lists (lengths match each other).
    # These may be mutated by passthrough-mode handling on the controller,
    # so they are stored here after that resolution step.
    illumination_sources: List[str]
    illumination_intensities: List[float]
    exposures: List[float]
    gains: List[float]

    # Output destination
    exp_name: str
    dir_path: str
    file_name: str

    # Stage position captured once at run start
    initial_xyz: Dict[str, float]             # {"X": …, "Y": …, "Z": …}

    # Behavior toggles resolved before dispatch
    keep_illumination_on: bool
    is_rgb: bool

    # Per-iteration time index (mutated inside the time-lapse loop).
    timepoint_index: int = 0

    @property
    def initial_z_position(self) -> float:
        return self.initial_xyz.get("Z", 0.0)

    @property
    def n_times(self) -> int:
        return self.experiment.parameterValue.numberOfImages

    @property
    def t_period(self) -> float:
        return self.experiment.parameterValue.timeLapsePeriod

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------
    def to_kwargs(self) -> Dict[str, Any]:
        """Flatten the context to the legacy kwargs ``execute_experiment`` understands.

        Kept so the existing kwargs-driven body of the execution-mode
        classes does not need to be rewritten in lock-step with the
        controller refactor.  All values come from either ``self`` or the
        ``ParameterValue`` on the wrapped experiment – there is no
        independent state.
        """
        p = self.experiment.parameterValue
        return {
            "snake_tiles": self.snake_tiles,
            "illumination_intensities": self.illumination_intensities,
            "illumination_sources": self.illumination_sources,
            "z_positions": self.z_positions,
            "initial_z_position": self.initial_z_position,
            "exposures": self.exposures,
            "gains": self.gains,
            "exp_name": self.exp_name,
            "dir_path": self.dir_path,
            "m_file_name": self.file_name,
            "t": self.timepoint_index,
            "n_times": self.n_times,
            "is_auto_focus": p.autoFocus,
            "autofocus_min": p.autoFocusMin,
            "autofocus_max": p.autoFocusMax,
            "autofocus_step_size": p.autoFocusStepSize,
            "autofocus_illumination_channel": p.autoFocusIlluminationChannel or "",
            "autofocus_mode": p.autoFocusMode,
            "autofocus_software_method": p.autoFocusSoftwareMethod,
            "autofocus_hc_initial_step": p.autoFocusHillClimbingInitialStep,
            "autofocus_hc_min_step": p.autoFocusHillClimbingMinStep,
            "autofocus_hc_step_reduction": p.autoFocusHillClimbingStepReduction,
            "autofocus_hc_max_iterations": p.autoFocusHillClimbingMaxIterations,
            "autofocus_target_focus_setpoint": p.autofocus_target_focus_setpoint,
            "autofocus_max_attempts": p.autofocus_max_attempts,
            "t_period": self.t_period,
            "isRGB": self.is_rgb,
            "t_pre_s": p.performance_t_pre_s,
            "t_post_s": p.performance_t_post_s,
            "keep_illumination_on": self.keep_illumination_on,
        }

    def performance_experiment_params(self) -> Dict[str, Any]:
        """Build the legacy ``experiment_params`` dict for performance mode."""
        return {
            "mExperiment": self.experiment,
            "tPeriod": self.t_period,
            "nTimes": self.n_times,
        }


__all__ = ["ExecutionContext"]
