"""Focus-map measurement, fitting, persistence and the pre-scan mapping phase.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import time
from fastapi import HTTPException
from typing import List, Optional, Dict, Any, Tuple

import os

from imswitch.imcommon.model import APIExport

from imswitch.imcontrol.controller.controllers.experiment_controller import FocusMapConfig, FocusMapFromPointsRequest, ParameterValue
from imswitch.imcontrol.model.focus_map import FocusMap, fit_quality_text


def _sanitize_sample(sample: str) -> str:
    """Subfolder name for a sample's focus maps ("" -> the shared default)."""
    cleaned = "".join(c for c in (sample or "") if c.isalnum() or c in "-_")[:40]
    return cleaned or "default"


class FocusMapMixin:
    """Focus-map measurement, fitting, persistence and the pre-scan mapping phase."""

    def _focus_map_af_kwargs(self, parameterValue=None) -> Dict[str, Any]:
        """Autofocus settings for the focus-map phase.

        Taken from the experiment's own ParameterValue when there is one, so
        there is a single place in the app to configure autofocus. Falls back to
        the model defaults for a standalone computeFocusMap call.
        """
        if parameterValue is None:
            parameterValue = ParameterValue(
                timeLapsePeriod=0.0, numberOfImages=1, autoFocus=True,
                autoFocusMin=-100.0, autoFocusMax=100.0, autoFocusStepSize=10.0,
                zStack=False, zStackMin=0.0, zStackMax=0.0,
            )
        kwargs = parameterValue.autofocus_kwargs()
        # Mapping wants a general Z surface, not a perfect focus per point:
        # no Gaussian smoothing keeps it fast and avoids fit trouble at low SNR.
        kwargs["af_n_gauss"] = 0
        return kwargs

    def _run_focus_map_phase(self, areas, config: FocusMapConfig, af_kwargs=None):
        """
        Run focus mapping before the main acquisition loop.

        Measures focus on a grid within each region and fits a Z surface for it,
        stored under that region's own id. Called from startWellplateExperiment
        when focusMap.enabled == True.

        Args:
            areas: regions_to_areas() output — the same region ids the
                acquisition loop will look maps up by.
        """
        # Store config for channel_offsets access during acquisition
        self._focus_map_config = config
        af_kwargs = af_kwargs or self._focus_map_af_kwargs()
        self.focus_map_manager.clear_abort()

        if not areas:
            self._logger.warning("Focus map: no areas found – skipping")
            return

        # ----------------------------------------------------------
        # Option: reuse a pre-existing manual / global map for all
        # groups by interpolation instead of measuring a new grid.
        # ----------------------------------------------------------
        if config.use_manual_map:
            # Explicit manual points supplied for THIS run are the source of
            # truth.  Measure/fit them fresh under "manual" so that a map left
            # over from a previous acquisition — an automatic "global"/per-area
            # map, or an earlier "manual" fit made with a *different* set of
            # points — can never override the points the user just placed.
            # Without this, the stale "manual" map would be reused below and
            # the freshly supplied config.points silently ignored.
            if config.points:
                self._logger.info(
                    f"Focus map: (re)measuring {len(config.points)} supplied manual "
                    f"point(s) before the tiled scan (overriding any stale map)"
                )
                self.focus_map_manager.clear("manual")
                self._measure_focus_map_from_points(
                    config.points, config, af_kwargs=af_kwargs)
                source_fm = self.focus_map_manager.get("manual")
                # Drop stale per-area maps so the fresh manual surface is actually
                # interpolated onto every area below, instead of being shadowed by
                # the "already fitted" skip-guard in the loop.
                for area in areas:
                    self.focus_map_manager.clear(area["areaId"])
            else:
                # Reuse only the map the user explicitly fitted from points.
                # Falling back to "any fitted map" adopted stale surfaces from
                # a different sample.
                source_fm = self.focus_map_manager.get("manual")

            if source_fm is not None and source_fm.is_fitted:
                self._logger.info(
                    f"Focus map: reusing manual map [{source_fm.group_id}] "
                    f"for {len(areas)} group(s) via interpolation"
                )
                for area in areas:
                    gid = area["areaId"]
                    if self.focus_map_manager.has_fitted_map(gid):
                        self._logger.info(
                            f"Focus map [{gid}]: already fitted – skipping interpolation"
                        )
                        continue
                    try:
                        new_fm = source_fm.interpolate_to_region(
                            group_id=gid,
                            group_name=area["areaName"],
                            bounds=area["bounds"],
                            rows=max(config.rows, 3),
                            cols=max(config.cols, 3),
                            logger=self._logger,
                        )
                        self.focus_map_manager._maps[gid] = new_fm
                        self._logger.info(
                            f"Focus map [{gid}]: interpolated from [{source_fm.group_id}] "
                            f"({new_fm.n_points} pts, fitted={new_fm.is_fitted})"
                        )
                    except Exception as e:
                        self._logger.warning(
                            f"Focus map [{gid}]: interpolation from manual map failed ({e}), "
                            f"falling back to measurement"
                        )
                        self._compute_focus_map_for_group(
                            group_id=gid,
                            group_name=area["areaName"],
                            bounds=area["bounds"],
                            config=config,
                            af_kwargs=af_kwargs,
)
                return
            else:
                self._logger.warning(
                    "Focus map: use_manual_map is enabled but no fitted manual/global map found – "
                    "falling back to per-group measurement"
                )

        if config.fit_by_region:
            # Fit separately per region – skip groups that already have a valid fit.
            # Detect degenerate areas (single-FOV with zero-size bounds) and combine
            # them into a single global focus map instead of duplicating measurements.
            DEGEN_THRESHOLD = 1.0  # microns – area extent smaller than this is degenerate
            degenerate_areas = []
            normal_areas = []
            for area in areas:
                b = area["bounds"]
                w = abs(b["maxX"] - b["minX"])
                h = abs(b["maxY"] - b["minY"])
                if w < DEGEN_THRESHOLD and h < DEGEN_THRESHOLD:
                    degenerate_areas.append(area)
                else:
                    normal_areas.append(area)

            # Handle degenerate areas: use the combined bounds of ALL areas
            # so that the focus grid covers the exterior of the whole scan region.
            if degenerate_areas:
                all_areas_for_bounds = areas  # use all areas (normal + degenerate)
                combined_bounds = {
                    "minX": min(a["bounds"]["minX"] for a in all_areas_for_bounds),
                    "maxX": max(a["bounds"]["maxX"] for a in all_areas_for_bounds),
                    "minY": min(a["bounds"]["minY"] for a in all_areas_for_bounds),
                    "maxY": max(a["bounds"]["maxY"] for a in all_areas_for_bounds),
                }
                self._logger.info(
                    f"Focus map: {len(degenerate_areas)} degenerate (single-FOV) area(s) detected – "
                    f"computing combined map using global bounds instead of per-area"
                )
                combined_gid = "global_combined"
                if not self.focus_map_manager.has_fitted_map(combined_gid):
                    self._compute_focus_map_for_group(
                        group_id=combined_gid,
                        group_name="Combined (single-FOV areas)",
                        bounds=combined_bounds,
                        config=config,
                    )
                # Assign the combined map to each degenerate area
                combined_fm = self.focus_map_manager.get(combined_gid)
                if combined_fm is not None and combined_fm.is_fitted:
                    for area in degenerate_areas:
                        gid = area["areaId"]
                        self.focus_map_manager._maps[gid] = combined_fm
                        self._logger.info(
                            f"Focus map [{gid}]: assigned combined map (single-FOV area)"
                        )

            # Handle normal (non-degenerate) areas individually
            for area in normal_areas:
                gid = area["areaId"]
                if self.focus_map_manager.has_fitted_map(gid):
                    self._logger.info(
                        f"Focus map [{gid}]: using pre-computed map (already fitted)"
                    )
                    continue
                self._compute_focus_map_for_group(
                    group_id=gid,
                    group_name=area["areaName"],
                    bounds=area["bounds"],
                    config=config,
                            af_kwargs=af_kwargs,
)
        else:
            # Fit globally: merge all bounds
            all_min_x = min(a["bounds"]["minX"] for a in areas)
            all_max_x = max(a["bounds"]["maxX"] for a in areas)
            all_min_y = min(a["bounds"]["minY"] for a in areas)
            all_max_y = max(a["bounds"]["maxY"] for a in areas)
            merged = {
                "minX": all_min_x, "maxX": all_max_x,
                "minY": all_min_y, "maxY": all_max_y,
            }
            if self.focus_map_manager.has_fitted_map("global"):
                self._logger.info("Focus map [global]: using pre-computed map (already fitted)")
            else:
                self._compute_focus_map_for_group(
                    group_id="global",
                    group_name="Global Fit",
                    bounds=merged,
                    config=config,
                            af_kwargs=af_kwargs,
)

    # ================================================================
    # Focus Map API endpoints
    # ================================================================

    @APIExport(requestType="POST")
    def computeFocusMap(self, focusMapConfig: Optional[FocusMapConfig] = None,
                        group_id: Optional[str] = None):
        """
        Compute focus map for one or all scan groups.

        Runs autofocus on a grid of positions within each group's bounds,
        fits a Z surface, and stores the results for use during acquisition.

        Args:
            focusMapConfig: Override config (uses experiment default if None).
                            May include scan_areas from the frontend so that
                            the correct XY bounds are known even before an
                            experiment has been started.
            group_id: If provided, only compute for this group. Otherwise compute for all.

        Returns:
            Dict with focus map results per group
        """
        if focusMapConfig is None:
            focusMapConfig = FocusMapConfig(enabled=True)
        # Store config for channel_offsets access during acquisition
        self._focus_map_config = focusMapConfig
        # Autofocus settings come from the experiment's own ParameterValue —
        # the one place they are configured — with model defaults when this is
        # called standalone from the focus-map panel.
        af_kwargs = self._focus_map_af_kwargs(focusMapConfig.autofocus)

        # Clear any previous abort request
        self.focus_map_manager.clear_abort()

        self._logger.info(
            f"Computing focus map: method={focusMapConfig.method}, "
            f"grid={focusMapConfig.rows}x{focusMapConfig.cols}, "
            f"group_id={group_id}"
        )

        # Determine scan area bounds – prefer scan_areas passed in the config,
        # then fall back to last experiment areas, then to current stage pos.
        results = {}

        if focusMapConfig.scan_areas is not None and len(focusMapConfig.scan_areas) > 0:
            # Caller provided scan areas directly (e.g. from frontend)
            self._last_scan_areas = focusMapConfig.scan_areas

        effective_scan_areas = getattr(self, '_last_scan_areas', None)
        if effective_scan_areas is None or len(effective_scan_areas) == 0:
            # Fallback: use current stage position as single area
            pos = self.mStage.getPosition()
            effective_scan_areas = [{
                "areaId": "current",
                "areaName": "Current Position",
                "bounds": {
                    "minX": pos.get("X", 0) - 500,
                    "maxX": pos.get("X", 0) + 500,
                    "minY": pos.get("Y", 0) - 500,
                    "maxY": pos.get("Y", 0) + 500,
                },
            }]

        # When a curated grid (frontend preview) is supplied, it is the
        # authoritative plan for this call: areas whose points were all
        # removed by the user are skipped instead of re-measured.
        has_curated_grid = bool(focusMapConfig.grid_points)

        for area in effective_scan_areas:
            area_id = area.get("areaId", "default")
            area_name = area.get("areaName", area_id)

            if group_id is not None and area_id != group_id:
                continue

            if has_curated_grid and not self._explicit_grid_points_for_group(
                    focusMapConfig, area_id):
                self._logger.info(
                    f"Focus map [{area_id}]: all planned grid points removed by user – skipping"
                )
                continue

            bounds = area.get("bounds", {})
            result = self._compute_focus_map_for_group(
                group_id=area_id,
                group_name=area_name,
                bounds=bounds,
                config=focusMapConfig,
                af_kwargs=af_kwargs,
            )
            results[area_id] = result

        return results

    @staticmethod
    def _explicit_grid_points_for_group(config: FocusMapConfig,
                                        group_id: str) -> List[Tuple[float, float]]:
        """Return the user-curated grid positions belonging to ``group_id``.

        ``config.grid_points`` entries carry the frontend scan-area id as
        ``group_id``; entries without one apply to every group. Returns an
        empty list when nothing matches (caller falls back to generate_grid).
        """
        explicit = []
        for p in (config.grid_points or []):
            try:
                if p.get("group_id") not in (None, group_id):
                    continue
                explicit.append((float(p["x"]), float(p["y"])))
            except (TypeError, ValueError, KeyError):
                continue
        return explicit

    def _compute_focus_map_for_group(self, group_id: str, group_name: str,
                                     bounds: Dict[str, float],
                                     config: FocusMapConfig,
                                     af_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute focus map for a single group by moving stage and running autofocus.

        Args:
            group_id: Identifier for this scan group
            group_name: Human-readable name
            bounds: Dict with minX, maxX, minY, maxY
            config: FocusMapConfig

        Returns:
            FocusMapResult as dict
        """
        fm = self.focus_map_manager.get_or_create(
            group_id=group_id,
            group_name=group_name,
            method=config.method,
            smoothing_factor=config.smoothing_factor,
            z_offset=config.z_offset,
            clamp_enabled=config.clamp_enabled,
            z_min=config.z_min,
            z_max=config.z_max,
        )
        # Always clear points when explicitly (re)computing a focus map.
        # Precomputed maps are protected by the skip-logic in _run_focus_map_phase
        # and computeFocusMap; this function is only reached when we truly want
        # to measure from scratch.
        fm.clear_points()

        # User-curated grid positions from the frontend preview take precedence
        # over the generated grid (entries tagged with another group_id belong
        # to a different scan area and are ignored here).
        grid = self._explicit_grid_points_for_group(config, group_id)
        if grid:
            self._logger.info(
                f"Focus map [{group_id}]: using {len(grid)} user-curated grid point(s) "
                f"instead of the generated {config.rows}x{config.cols} grid"
            )
        else:
            grid = FocusMap.generate_grid(
                bounds=bounds,
                rows=config.rows,
                cols=config.cols,
                add_margin=config.add_margin,
            )
        af_kwargs = af_kwargs or self._focus_map_af_kwargs()
        self._logger.info(f"Focus map [{group_id}]: measuring {len(grid)} points")

        # Measure each grid point
        for i, (gx, gy) in enumerate(grid):
            # Check abort flag
            if self.focus_map_manager.abort_requested:
                self._logger.warning(f"Focus map [{group_id}]: aborted by user at point {i+1}/{len(grid)}")
                break
            try:
                # Move stage to measurement position
                self.move_stage_xy(posX=gx, posY=gy, relative=False)
                time.sleep(0.1)  # Settle time

                best_z = self.autofocus(**af_kwargs)

                if best_z is None:
                    # SKIP the point. Using the stage position instead would
                    # plant the scan-range start into the surface and tilt the
                    # whole map; a map fitted from fewer, real points is honest
                    # about what it knows (see fit stats).
                    self._logger.warning(
                        f"Focus map [{group_id}]: autofocus failed at "
                        f"({gx:.1f}, {gy:.1f}) — skipping this point"
                    )
                    continue

                fm.add_point(gx, gy, float(best_z))
                self._logger.debug(
                    f"Focus map [{group_id}]: point {i+1}/{len(grid)} "
                    f"at ({gx:.1f}, {gy:.1f}) → Z={best_z:.3f}"
                )
                # settle for a moment
                time.sleep(0.5)

            except Exception as e:
                self._logger.error(f"Focus map [{group_id}]: failed at ({gx:.1f}, {gy:.1f}): {e}")

        # Fit the surface
        try:
            stats = fm.fit()
            self._logger.info(
                f"Focus map [{group_id}]: fitted {stats.method}, "
                f"n={stats.n_points}, {fit_quality_text(stats)}"
            )
        except ValueError as e:
            self._logger.error(f"Focus map [{group_id}]: fit failed: {e}")
            return fm.to_result().to_dict()

        # Save artifacts if requested
        if config.store_debug_artifacts:
            try:
                save_dir = os.path.join(self.save_dir, "focus_maps")
                fm.save(save_dir)
            except Exception as e:
                self._logger.warning(f"Failed to save focus map artifacts: {e}")

        return fm.to_result().to_dict()

    def _measure_focus_map_from_points(self, points: List[Dict[str, Any]],
                                       config: "FocusMapConfig",
                                       group_id: str = "manual",
                                       group_name: str = "Manual Points",
                                       af_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Drive to each manual XY point, autofocus to MEASURE Z, then fit.

        Unlike ``computeFocusMapFromPoints`` (which fits already-known XYZ), this
        measures Z at each point on the backend (blocking). A point that already
        carries a numeric ``z`` is used as-is (autofocus skipped). The fitted
        surface is stored under ``group_id`` so it is usable via use_manual_map.
        """
        fm = self.focus_map_manager.get_or_create(
            group_id=group_id, group_name=group_name,
            method=config.method, smoothing_factor=config.smoothing_factor,
            z_offset=config.z_offset, clamp_enabled=config.clamp_enabled,
            z_min=config.z_min, z_max=config.z_max,
        )
        fm.clear_points()
        af_kwargs = af_kwargs or self._focus_map_af_kwargs()
        self._logger.info(f"Focus map [{group_id}]: measuring {len(points)} manual point(s)")

        for i, pt in enumerate(points):
            if self.focus_map_manager.abort_requested:
                self._logger.warning(f"Focus map [{group_id}]: aborted at point {i+1}/{len(points)}")
                break
            try:
                gx = float(pt.get("x"))
                gy = float(pt.get("y"))
            except (TypeError, ValueError):
                continue
            z_in = pt.get("z", None)
            try:
                if z_in is not None:
                    best_z = float(z_in)  # user-provided Z; no autofocus needed
                else:
                    self.move_stage_xy(posX=gx, posY=gy, relative=False)
                    time.sleep(0.4)  # settle
                    best_z = self.autofocus(**af_kwargs)
                    if best_z is None:
                        self._logger.warning(
                            f"Focus map [{group_id}]: autofocus failed at "
                            f"({gx:.1f}, {gy:.1f}) — skipping this point"
                        )
                        continue
                fm.add_point(gx, gy, float(best_z))
                time.sleep(0.3)
            except Exception as e:
                self._logger.error(f"Focus map [{group_id}]: failed at ({gx:.1f}, {gy:.1f}): {e}")

        try:
            stats = fm.fit()
            self._logger.info(
                f"Focus map [{group_id}]: measured+fitted {stats.method}, "
                f"n={stats.n_points}, {fit_quality_text(stats)}"
            )
        except ValueError as e:
            self._logger.error(f"Focus map [{group_id}]: fit failed: {e}")
        return fm.to_result().to_dict()

    @APIExport(requestType="POST")
    def measureFocusMapFromPoints(self, focusMapConfig: Optional[FocusMapConfig] = None):
        """Autofocus-measure Z at each manual point (focusMapConfig.points) and fit.

        Blocking single call (the frontend awaits it) so the measurement runs and
        completes on the backend rather than looping per-point from the browser.
        Stores the fitted map under "manual"; enable "Use manual map for all
        groups" to apply it during acquisition.
        """
        if focusMapConfig is None:
            focusMapConfig = FocusMapConfig(enabled=True)
        points = focusMapConfig.points or []
        if not points:
            return {"error": "No manual points provided (focusMapConfig.points is empty)"}
        self._focus_map_config = focusMapConfig
        self.focus_map_manager.clear_abort()
        result = self._measure_focus_map_from_points(
            points, focusMapConfig,
            af_kwargs=self._focus_map_af_kwargs(focusMapConfig.autofocus),
        )
        return {"manual": result}

    @APIExport(requestType="GET")
    def getFocusMap(self, group_id: Optional[str] = None):
        """
        Get saved focus maps.

        Args:
            group_id: If provided, get only this group. Otherwise get all.

        Returns:
            Focus map data per group
        """
        if group_id is not None:
            fm = self.focus_map_manager.get(group_id)
            if fm is None:
                raise HTTPException(status_code=404, detail=f"No focus map for group '{group_id}'")
            return fm.to_result().to_dict()
        return self.focus_map_manager.to_dict()

    @APIExport(requestType="GET")
    def getFocusMapSummary(self):
        """One line per region: which map a run would use, and how good it is.

        Answers "which map is this run using?" without opening the focus panel.
        """
        regions = []
        for area in (getattr(self, "_last_scan_areas", None) or []):
            fm = self.focus_map_manager.get(area["areaId"])
            stats = fm.fit_stats if (fm is not None and fm.is_fitted) else None
            regions.append({
                "region_id": area["areaId"],
                "region_name": area["areaName"],
                "has_map": stats is not None,
                "method": stats.method if stats else None,
                "n_points": stats.n_points if stats else 0,
                "z_offset_um": stats.z_offset if stats else 0.0,
                "quality": fit_quality_text(stats) if stats else "no map — per-point Z will be used",
                "reason": stats.fallback_reason if stats else "",
            })
        return {
            "regions": regions,
            "focus_map_active": bool(getattr(self, "_focus_map_active", False)),
            # The other additive Z corrections, so their sum is visible in one
            # place instead of being spread over four independent knobs.
            "channel_offsets_um": (
                getattr(getattr(self, "_focus_map_config", None), "channel_offsets", None) or {}
            ),
            "autofocus_offsets_um": dict(self._experiment_af_offsets),
        }

    @APIExport(requestType="POST")
    def getFocusMapPreview(self, group_id: str, resolution: int = 30):
        """
        Get a preview grid for visualization of a focus map.

        Args:
            group_id: Group to preview
            resolution: Grid resolution for preview

        Returns:
            Dict with x, y, z arrays and raw points
        """
        fm = self.focus_map_manager.get(group_id)
        if fm is None or not fm.is_fitted:
            raise HTTPException(status_code=404, detail=f"No fitted focus map for group '{group_id}'")

        return fm.generate_preview_grid(resolution=resolution)

    @APIExport(requestType="POST")
    def clearFocusMap(self, group_id: Optional[str] = None):
        """
        Clear focus map(s).

        Args:
            group_id: If provided, clear only this group. Otherwise clear all.

        Returns:
            Status message
        """
        self.focus_map_manager.clear(group_id)
        target = f"group '{group_id}'" if group_id else "all groups"
        self._logger.info(f"Cleared focus map for {target}")
        return {"status": "cleared", "target": target}

    @APIExport(requestType="POST")
    def interruptFocusMap(self):
        """
        Interrupt an ongoing focus map computation.

        Sets the abort flag so the measurement loop stops at the next grid point.
        The map that was being computed will still try to fit with whatever
        points have been collected so far.

        Returns:
            Status message
        """
        self.focus_map_manager.request_abort()
        self._logger.info("Focus map computation interrupt requested")
        return {"status": "interrupt_requested"}

    @APIExport(requestType="GET")
    def saveFocusMaps(self, path: str = "", sample: str = ""):
        """
        Save all current focus maps to disk as JSON files.

        Args:
            path: Directory to save into. Empty string uses default location.
            sample: Optional sample name. Map sets are kept in their own
                subfolder per sample; one shared folder made it impossible to
                tell whose surface a stored map belonged to.

        Returns:
            Dict with saved file info
        """
        if not path:
            path = os.path.join(self.save_dir, "focus_maps", _sanitize_sample(sample))
        os.makedirs(path, exist_ok=True)
        saved = self.focus_map_manager.save_all(path)
        self._logger.info(f"Saved {len(saved)} focus maps to {path}")
        return {"saved_files": saved, "count": len(saved), "path": path}

    @APIExport(requestType="GET")
    def loadFocusMaps(self, path: str = "", sample: str = ""):
        """
        Load focus maps from disk, REPLACING the maps currently in memory.

        Args:
            path: Directory to load from. Empty string uses default location.
            sample: Optional sample name, matching saveFocusMaps.

        Returns:
            Dict with loaded map info
        """
        if not path:
            path = os.path.join(self.save_dir, "focus_maps", _sanitize_sample(sample))
        if not os.path.isdir(path):
            return {"loaded_count": 0, "path": path, "error": "Directory not found"}
        count = self.focus_map_manager.load_all(path)
        self._logger.info(f"Loaded {count} focus maps from {path}")
        maps_dict = self.focus_map_manager.to_dict()
        return {"loaded_count": count, "path": path, "maps": maps_dict}

    @APIExport(requestType="POST")
    def computeFocusMapFromPoints(self, request: FocusMapFromPointsRequest):
        """
        Compute a focus map from manually specified XYZ points.

        Instead of running autofocus on a grid, the user provides pre-measured
        reference points (e.g. from clicking on the wellplate viewer and
        recording the current stage Z).

        Args:
            request: FocusMapFromPointsRequest with points and fit config

        Returns:
            FocusMapResult as dict
        """
        points = request.points
        group_id = request.group_id
        group_name = request.group_name
        method = request.method
        smoothing_factor = request.smoothing_factor
        z_offset = request.z_offset
        clamp_enabled = request.clamp_enabled
        z_min = request.z_min
        z_max = request.z_max

        if not points or len(points) < 1:
            return {"error": "At least 1 point is required"}

        self._logger.info(
            f"Computing focus map from {len(points)} manual points, "
            f"method={method}, group_id={group_id}"
        )

        fm = self.focus_map_manager.get_or_create(
            group_id=group_id,
            group_name=group_name,
            method=method,
            smoothing_factor=smoothing_factor,
            z_offset=z_offset,
            clamp_enabled=clamp_enabled,
            z_min=z_min,
            z_max=z_max,
        )
        fm.clear_points()

        # Add all manual points
        for pt in points:
            x = pt.get("x", 0)
            y = pt.get("y", 0)
            z = pt.get("z", 0)
            fm.add_point(float(x), float(y), float(z))

        # Fit the surface
        try:
            stats = fm.fit()
            self._logger.info(
                f"Focus map [{group_id}]: fitted {stats.method} from manual points, "
                f"n={stats.n_points}, {fit_quality_text(stats)}"
            )
        except ValueError as e:
            self._logger.error(f"Focus map [{group_id}]: fit failed: {e}")
            return fm.to_result().to_dict()

        return fm.to_result().to_dict()

    def apply_focus_map_z(self, x: float, y: float, group_id: str = "default",
                          channel: Optional[str] = None) -> Optional[float]:
        """
        Get the focus-mapped Z for a given XY position during acquisition.

        This is called by the normal mode experiment execution to adjust Z
        before acquiring at each tile position.

        Args:
            x: X position
            y: Y position
            group_id: Scan group identifier
            channel: Optional illumination channel name. If provided and
                     channel_offsets are configured in the FocusMapConfig,
                     the per-channel offset will be added to the result.

        Returns:
            Estimated Z position, or None if no map available
        """
        z = self.focus_map_manager.interpolate(x, y, group_id)
        if z is not None and channel:
            # Apply per-channel Z offset if configured
            focus_map_config = getattr(self, '_focus_map_config', None)
            if focus_map_config and focus_map_config.channel_offsets:
                channel_offset = focus_map_config.channel_offsets.get(channel, 0.0)
                if channel_offset != 0.0:
                    self._logger.debug(
                        f"Applying channel offset for '{channel}': {channel_offset} µm"
                    )
                    z += channel_offset
        return z

    # ── Overview Camera Registration Endpoints ─────────────────────────────────
