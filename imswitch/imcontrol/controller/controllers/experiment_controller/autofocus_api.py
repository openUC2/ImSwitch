"""Hardware and software autofocus, and the per-region Z offsets they produce.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import time
from typing import Optional


class AutofocusMixin:
    """Hardware and software autofocus, and the per-region Z offsets they produce."""

    def autofocus_hardware(self, target_focus_setpoint: Optional[float] = None, max_attempts=2, illuminationChannel: str = "") -> Optional[float]:
        """Perform hardware-based one-shot autofocus using FocusLockController.

        This is significantly faster than software autofocus because it:
        - Captures only ONE frame from dedicated autofocus camera
        - Uses pre-calibrated linear relationship (focus metric → Z position)
        - No Z-sweep required

        Args:
            illuminationChannel: Selected illumination channel for autofocus (currently unused)

        Returns:
            float: Best focus Z position in µm, or None if autofocus failed
        """
        self._logger.debug("Performing hardware-based one-shot autofocus...")

        # Get the focus lock controller
        try:
            focusLockController = self._master.getController('FocusLock')
        except Exception as e:
            self._logger.warning(f"FocusLockController not available: {e}")
            return None

        if focusLockController is None:
            self._logger.warning("FocusLockController not available - skipping hardware autofocus")
            return None

        # Check if calibration exists
        try:
            calib_status = focusLockController.getCalibrationStatus()
            if not calib_status.get('calibrated', False):
                self._logger.error("Hardware autofocus requires calibration. Please run focus calibration first.")
                return None
        except Exception as e:
            self._logger.error(f"Failed to check calibration status: {e}")
            return None

        # Perform one-shot autofocus
        try:
            result = focusLockController.performOneStepAutofocus(
                target_focus_setpoint=target_focus_setpoint,
                move_to_focus=True,
                max_attempts=max_attempts,
                threshold_um=0.5,
                in_background=False
            )
            if result.get('success', False):
                target_z = result.get('z_offset')
                self._logger.info(
                    f"Hardware autofocus successful: "
                    f"Z={target_z:.2f}µm, error={result.get('final_error_um', 0):.3f}µm, "
                    f"attempts={result.get('num_attempts', 0)}"
                )
                return target_z
            else:
                error_msg = result.get('error', 'Unknown error')
                self._logger.error(f"Hardware autofocus failed: {error_msg}")
                return None

        except Exception as e:
            self._logger.error(f"Hardware autofocus exception: {e}")
            return None

    def autofocus(self, minZ: float=0, maxZ: float=0, stepSize: float=0,
                  illuminationChannel: str="", mode: str="software",
                  max_attempts: int=2,
                  target_focus_setpoint: Optional[float] = None,
                  af_range: float = 100.0,
                  af_resolution: float = 10.0,
                  af_cropsize: int = 2048,
                  af_algorithm: str = "LAPE",
                  af_settle_time: float = 0.1,
                  af_static_offset: float = 0.0,
                  af_two_stage: bool = False,
                  af_n_gauss: int = 0,
                  af_software_method: str = "scan",
                  af_hc_initial_step: float = 20.0,
                  af_hc_min_step: float = 1.0,
                  af_hc_step_reduction: float = 0.5,
                  af_hc_max_iterations: int = 50) -> Optional[float]:
        """Perform autofocus using either hardware or software method.

        Args:
            minZ: Legacy minimum Z position (overridden by af_range)
            maxZ: Legacy maximum Z position (overridden by af_range)
            stepSize: Legacy step size (overridden by af_resolution)
            illuminationChannel: Selected illumination channel for autofocus
            mode: "hardware" (fast, one-shot) or "software" (slow, Z-sweep)
            max_attempts: Max retry attempts for hardware AF
            target_focus_setpoint: Target setpoint for hardware AF
            af_range: Autofocus Z range ±µm from current Z
            af_resolution: Z step size for autofocus
            af_cropsize: Crop size for focus algorithm
            af_algorithm: Focus quality algorithm (LAPE, GLVA, JPEG)
            af_settle_time: Settle time in seconds
            af_static_offset: Static Z offset after autofocus
            af_two_stage: Use two-stage (coarse+fine) autofocus
            af_n_gauss: Gaussian kernel size

        Returns:
            float: Best focus Z position, or None if autofocus failed
        """
        self._logger.debug(
            f"Performing autofocus (mode={mode}) with parameters "
            f"af_range={af_range}, af_resolution={af_resolution}, "
            f"af_algorithm={af_algorithm}, channel={illuminationChannel}"
        )

        # Route to appropriate autofocus method
        if mode == "hardware":
            return self.autofocus_hardware(target_focus_setpoint=target_focus_setpoint,
                                           max_attempts=max_attempts,
                                           illuminationChannel=illuminationChannel)
        else:
            return self.autofocus_software(
                af_range=af_range,
                af_resolution=af_resolution,
                af_cropsize=af_cropsize,
                af_algorithm=af_algorithm,
                af_settle_time=af_settle_time,
                af_static_offset=af_static_offset,
                af_two_stage=af_two_stage,
                af_n_gauss=af_n_gauss,
                illuminationChannel=illuminationChannel,
                minZ=minZ,
                maxZ=maxZ,
                stepSize=stepSize,
                af_software_method=af_software_method,
                af_hc_initial_step=af_hc_initial_step,
                af_hc_min_step=af_hc_min_step,
                af_hc_step_reduction=af_hc_step_reduction,
                af_hc_max_iterations=af_hc_max_iterations,
            )

    @staticmethod
    def _autofocus_timeout_s(af_range: float, af_resolution: float,
                             af_settle_time: float, two_stage: bool) -> float:
        """Budget for one autofocus scan: steps x (settle + a frame), 3x safety.
        A flat two minutes made every hang cost the ceiling.
        """
        steps = max(1, int(2 * abs(af_range) / max(abs(af_resolution), 1e-6)) + 1)   # ±af_range
        per_step_s = max(af_settle_time, 0.0) + 1.0
        budget = 3.0 * steps * per_step_s * (2.0 if two_stage else 1.0)
        return min(120.0, max(10.0, budget))

    def _light_for_autofocus(self, channel: str = "") -> str:
        """Switch on the light autofocus looks through; returns its name ("" = none).

        Without a channel of its own, autofocus uses the run's first light
        that is on (at the run's intensity), and none in a passthrough run. The focus-map phase runs before any tile
        has switched a light on, and focusing on dark frames rejected nearly
        every point of a FRAME slide scan (the rest were noise picks).
        """
        sources = list(getattr(self, "_illuminationSources", None) or [])
        intensities = list(getattr(self, "_illuminationIntensities", None) or [])
        if not channel:
            if getattr(self, "_illuminationPassthrough", False):
                return ""          # lit by hand: the run's power 1 is only a sentinel
            names = list(getattr(self, "allIlluNames", None) or [])
            channel = next((s for s, v in zip(sources, intensities)
                            if s in names and v and v > 0), "")
        lasers = getattr(getattr(self, "_master", None), "lasersManager", None)
        if not channel or lasers is None:
            return ""
        try:
            self._logger.debug(f"Activating illumination channel '{channel}' for autofocus")
            # The run's intensity for this light, by the run's own source list
            # (the device list has another order when lights are skipped)
            index = sources.index(channel) if channel in sources else -1
            power = (intensities[index] or 0) if 0 <= index < len(intensities) else 0
            if power <= 0:
                power = lasers[channel].getValue() or 50
            self.set_laser_power(power=power, channel=channel)
            return channel
        except Exception as e:
            self._logger.warning(f"Failed to set illumination channel {channel}: {e}")
            return ""

    def autofocus_software(self, af_range: float = 100.0, af_resolution: float = 10.0,
                           af_cropsize: int = 2048, af_algorithm: str = "LAPE",
                           af_settle_time: float = 0.1, af_static_offset: float = 0.0,
                           af_two_stage: bool = False, af_n_gauss: int = 0,
                           illuminationChannel: str = "",
                           minZ: float = 0, maxZ: float = 0, stepSize: float = 0,
                           af_software_method: str = "scan",
                           af_hc_initial_step: float = 20.0,
                           af_hc_min_step: float = 1.0,
                           af_hc_step_reduction: float = 0.5,
                           af_hc_max_iterations: int = 50):
        """Perform software-based autofocus using AutofocusController.

        Supports two methods:
        - 'scan' (Z-sweep): scans through Z range and fits Gaussian to find peak
        - 'hillClimbing': iterative gradient ascent for faster convergence

        All parameters are passed through from FocusMapConfig or experiment settings.
        Legacy minZ/maxZ/stepSize params are kept for backward compatibility but
        are overridden by af_range/af_resolution when those are non-default.

        Returns:
            float: Best focus Z position, or None if autofocus failed
        """
        self._logger.debug(
            "Performing software autofocus (method=%s)... "
            "af_range=%s, af_resolution=%s, af_algorithm=%s, af_cropsize=%s, "
            "af_settle_time=%s, af_n_gauss=%s, af_two_stage=%s, illumination=%s",
            af_software_method, af_range, af_resolution, af_algorithm, af_cropsize,
            af_settle_time, af_n_gauss, af_two_stage, illuminationChannel
        )

        # Get the autofocus controller
        autofocusController = self._master.getController('Autofocus')

        if autofocusController is None:
            self._logger.warning("AutofocusController not available - skipping autofocus")
            return None

        self._light_for_autofocus(illuminationChannel)

        af_started_at = time.time()
        try:
            if af_software_method == "hillClimbing":
                # Hill-climbing autofocus: iterative gradient ascent
                self._logger.debug(
                    "Using hill-climbing AF: initial_step=%s, min_step=%s, "
                    "step_reduction=%s, max_iterations=%s",
                    af_hc_initial_step, af_hc_min_step,
                    af_hc_step_reduction, af_hc_max_iterations
                )
                autofocusController.autoFocusHillClimbing(
                    initial_step=af_hc_initial_step,
                    min_step=af_hc_min_step,
                    step_reduction=af_hc_step_reduction,
                    max_iterations=af_hc_max_iterations,
                    tSettle=af_settle_time,
                    nCropsize=af_cropsize,
                    focusAlgorithm=af_algorithm,
                    nGauss=af_n_gauss,
                    static_offset=af_static_offset,
                )
            else:
                # Z-sweep autofocus: scan through range and fit Gaussian.
                # af_range is ±µm around the current Z; AutofocusController's
                # rangez is the FULL width (it scans ±rangez/2). Passing af_range
                # through scanned half the configured window (±50 µm for ±100).
                if af_range > 0:
                    rangez = 2.0 * af_range
                elif maxZ > minZ:
                    rangez = abs(maxZ - minZ)
                else:
                    rangez = 100.0

                # Determine resolution: prefer af_resolution, fall back to legacy stepSize
                resolutionz = af_resolution if af_resolution > 0 else (stepSize if stepSize > 0 else 10.0)

                autofocusController.autoFocus(
                    rangez=rangez,
                    resolutionz=resolutionz,
                    defocusz=0,
                    tSettle=af_settle_time,
                    isDebug=False,
                    nGauss=af_n_gauss,
                    nCropsize=af_cropsize,
                    focusAlgorithm=af_algorithm,
                    static_offset=af_static_offset,
                    twoStage=af_two_stage
                )

            # Wait for the autofocus thread (it runs in _AutofocusThead), but
            # only for as long as this scan could plausibly take. A flat 120 s
            # meant a hung autofocus cost two minutes at every single point.
            af_thread = getattr(autofocusController, '_AutofocusThead', None)
            if af_thread is not None and af_thread.is_alive():
                af_thread.join(timeout=self._autofocus_timeout_s(
                    af_range, af_resolution, af_settle_time, af_two_stage
                ))

            # Autofocus's own result, never the stage position: on a failure
            # or timed-out join the stage sits at the start of the scan range,
            # and feeding that into the Z offset defocused the rest of the run.
            result = getattr(autofocusController, '_lastAutofocusZ', None)
            elapsed = time.time() - af_started_at
            if result is None:
                self._logger.warning(
                    f"Autofocus produced no usable Z after {elapsed:.1f}s "
                    f"(alive={af_thread.is_alive() if af_thread else False})"
                )
                return None

            self._logger.info(f"Autofocus settled at Z={result:.2f} in {elapsed:.1f}s")
            return result

        except Exception as e:
            self._logger.error(f"Autofocus failed: {e}")
            return None

    def update_af_offset(self, context=None, metadata=None, expected_z: float = 0.0,
                         apply_global_offset: bool = True, region_id: str = "",
                         **kwargs):
        """Post-func for the Autofocus workflow step: store the region's Z offset.

        Records ``metadata["result"] - expected_z`` as this region's offset;
        capture Z-moves add it via ``move_stage_z(af_region_id=...)``.
        A failed autofocus returns None, leaves the offset untouched, and is
        counted — the run continues at the region's unmodified base Z.
        """
        if not apply_global_offset:
            return None
        measured = (metadata or {}).get("result")
        if measured is None:
            self._af_failures += 1
            self._logger.warning(
                f"Autofocus failed for region [{region_id}] "
                f"({self._af_failures} failure(s) so far); keeping its previous "
                "Z offset and continuing."
            )
            return None
        try:
            offset = float(measured) - float(expected_z)
            self._experiment_af_offsets[region_id] = offset
            self._af_successes += 1
            self._logger.info(
                f"Autofocus Z offset for region [{region_id}] updated to {offset:+.2f} µm "
                f"(measured={float(measured):.2f}, expected base={float(expected_z):.2f})"
            )
            return offset
        except Exception as e:
            self._af_failures += 1
            self._logger.debug(f"Could not update autofocus Z offset: {e}")
            return None
