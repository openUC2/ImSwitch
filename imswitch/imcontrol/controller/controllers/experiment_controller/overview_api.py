"""Overview-camera slide registration and the autonomous overview scan.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import time
from fastapi import HTTPException
import numpy as np
from typing import Optional, Dict, Any

import threading

from imswitch.imcommon.model import APIExport

from imswitch.imcontrol.model.overview_registration import OverviewRegistrationService, PixelPoint, StagePoint
from imswitch.imcontrol.model import configfiletools


class OverviewRegistrationMixin:
    """Overview-camera slide registration and the autonomous overview scan."""

    @APIExport(requestType="POST")
    def getOverviewRegistrationConfig(self, layout_data: dict = None, layout_name: str = "Heidstar 4x Histosample"):
        """
        Get overview wizard configuration with slot definitions for a layout.

        The frontend should POST its current wellLayout object (including any
        offsets already applied) so that the slot corners returned here match
        exactly what the WellSelector canvas renders.

        Args:
            layout_data: Full layout dict from the frontend (preferred).
                         Must contain at least 'name', 'wells' (list).
            layout_name: Fallback layout name used only when layout_data is None.

        Returns:
            Layout name, slot list with stage corners, corner convention,
            camera availability, and saved status per slide.
        """
        # 1. Prefer the layout sent by the frontend (already has offsets applied)
        if layout_data is not None and isinstance(layout_data, dict) and layout_data.get("wells"):
            layout_dict = layout_data
            layout_name = layout_data.get("name", layout_name)
        else:
            # 2. Try the labware manager (Opentrons defs by load_name)
            layout_dict = None
            if self.labware_manager:
                try:
                    lab = self.labware_manager.get(layout_name)
                    layout_dict = self._labware_to_layout_dict(lab)
                except KeyError:
                    layout_dict = None
            if layout_dict is None:
                # 3. Last-resort hardcoded Heidstar fallback
                self._logger.warning(
                    "No layout_data from frontend and labware lookup failed. "
                    "Using hardcoded Heidstar fallback – coordinates may not match canvas!"
                )
                layout_dict = {
                    "name": "Heidstar 4x Histosample",
                    "unit": "um",
                    "width": 127000,
                    "height": 84000,
                    "wells": [
                        {"x": 18400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide1"},
                        {"x": 48400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide2"},
                        {"x": 78400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide3"},
                        {"x": 108400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide4"},
                    ],
                }

        slots = self._overview_registration.get_slot_definitions(layout_dict)
        camera_name = self._overview_camera_name or "overviewcamera"
        status = self._overview_registration.get_status(camera_name, layout_name)

        return {
            "layoutName": layout_dict.get("name", layout_name),
            "cameraName": camera_name,
            "cameraAvailable": self._overview_camera is not None,
            "cornerConvention": "TL,TR,BR,BL",
            "cornerLabels": ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"],
            "slots": [s.model_dump() if hasattr(s, 'model_dump') else s.dict() for s in slots],
            "status": status.get("slides", {}),
        }

    @APIExport(requestType="POST")
    def snapOverviewImage(self, slot_id: str = "1", camera_name: str = ""):
        """
        Snap a single image from the overview camera for a given slot.

        Args:
            slot_id:     Slide slot index ("1" to "4")
            camera_name: Camera name (auto-detected if empty)

        Returns:
            Snapshot metadata including base64-encoded image.
        """
        cam = self._overview_camera
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"

        if cam is None:
            raise HTTPException(status_code=400, detail="Overview camera not available")

        frame = cam.getLatestFrame()
        if frame is None:
            raise HTTPException(status_code=500, detail="No frame from overview camera")


        # Get current stage position for traceability
        stage_x, stage_y, stage_z = 0.0, 0.0, 0.0
        try:
            if self.mStage is not None:
                pos = self.mStage.getPosition()
                stage_x = pos.get("X", 0.0)
                stage_y = pos.get("Y", 0.0)
                stage_z = pos.get("Z", 0.0)
        except Exception:
            pass

        # Make contiguous copy
        frame = np.ascontiguousarray(frame)

        # Save snapshot
        meta = self._overview_registration.save_snapshot(
            camera_name=cam_name,
            layout_name="current",
            slot_id=slot_id,
            image=frame,
            stage_x=stage_x,
            stage_y=stage_y,
            stage_z=stage_z,
        )

        # Encode as base64 JPEG for frontend display
        import cv2
        if len(frame.shape) == 2:
            encode_frame = frame
        else:
            encode_frame = frame
        _, jpg_buf = cv2.imencode(".jpg", encode_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        import base64
        b64_image = base64.b64encode(jpg_buf.tobytes()).decode("ascii")

        meta["imageBase64"] = b64_image
        meta["imageMimeType"] = "image/jpeg"
        meta["stagePosition"] = {"x": stage_x, "y": stage_y, "z": stage_z}
        return meta

    @APIExport(requestType="POST")
    def recaptureSlot(
        self,
        slot_id: str = "1",
        layout_data: dict = None,
        layout_name: str = "Heidstar 4x Histosample",
        camera_name: str = "",
    ):
        """Recapture a single overview slide slot instead of all 4.

        Returns immediately with ``{started: True, task: ...}``. The actual
        stage move + snap runs in a background thread; clients poll
        :meth:`getOverviewAsyncStatus` for completion / result. This unblocks
        the FastAPI worker thread so the rest of the UI stays responsive.
        """
        if not self._tryStartOverviewAsync(f"recapture_slot_{slot_id}"):
            return {
                "started": False,
                "error": "Another overview task is still running",
                "status": self._overview_async_status,
            }

        thread = threading.Thread(
            target=self._recaptureSlotWorker,
            args=(slot_id, layout_data, layout_name, camera_name),
            daemon=True,
            name=f"recaptureSlot-{slot_id}",
        )
        with self._overview_async_lock:
            self._overview_async_thread = thread
        thread.start()
        return {"started": True, "task": f"recapture_slot_{slot_id}"}

    def _recaptureSlotWorker(
        self,
        slot_id: str,
        layout_data: Optional[dict],
        layout_name: str,
        camera_name: str,
    ) -> None:
        try:
            result = self._recaptureSlotBlocking(
                slot_id=slot_id,
                layout_data=layout_data,
                layout_name=layout_name,
                camera_name=camera_name,
            )
            self._finishOverviewAsync(result=result)
        except Exception as exc:
            self._logger.error(f"recaptureSlot worker failed: {exc}", exc_info=True)
            self._finishOverviewAsync(error=str(exc))

    def _recaptureSlotBlocking(
        self,
        slot_id: str = "1",
        layout_data: dict = None,
        layout_name: str = "Heidstar 4x Histosample",
        camera_name: str = "",
    ) -> dict:
        """Synchronous body of recaptureSlot - executed inside the worker."""
        # Resolve the layout the same way getOverviewRegistrationConfig does.
        if layout_data is not None and isinstance(layout_data, dict) and layout_data.get("wells"):
            layout_dict = layout_data
            layout_name = layout_data.get("name", layout_name)
        else:
            layout_dict = None
            if self.labware_manager:
                try:
                    lab = self.labware_manager.get(layout_name)
                    layout_dict = self._labware_to_layout_dict(lab)
                except KeyError:
                    layout_dict = None
            if layout_dict is None:
                # Heidstar fallback (matches getOverviewRegistrationConfig).
                layout_dict = {
                    "name": "Heidstar 4x Histosample",
                    "unit": "um",
                    "width": 127000,
                    "height": 84000,
                    "wells": [
                        {"x": 18400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide1"},
                        {"x": 48400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide2"},
                        {"x": 78400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide3"},
                        {"x": 108400, "y": 40600, "shape": "rectangle", "width": 27000, "height": 74000, "name": "Slide4"},
                    ],
                }

        # Compute slot center from slot definitions.
        slots = self._overview_registration.get_slot_definitions(layout_dict)
        slot_index = None
        try:
            slot_index = int(slot_id) - 1  # API uses 1-based slot ids
        except (TypeError, ValueError):
            pass
        if slot_index is None or slot_index < 0 or slot_index >= len(slots):
            raise ValueError(
                f"Invalid slot_id '{slot_id}' for layout with {len(slots)} slots"
            )

        slot = slots[slot_index]
        slot_dict = slot.model_dump() if hasattr(slot, "model_dump") else slot.dict()
        center_x = slot_dict.get("centerX")
        center_y = slot_dict.get("centerY")
        if center_x is None or center_y is None:
            corners = (
                slot_dict.get("stageCorners")
                or slot_dict.get("corners")
                or []
            )
            if len(corners) >= 1:
                xs = [c.get("x", 0.0) for c in corners]
                ys = [c.get("y", 0.0) for c in corners]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
            else:
                well = layout_dict["wells"][slot_index]
                center_x = float(well.get("x", 0.0))
                center_y = float(well.get("y", 0.0))

        # Move the stage to the slot center (blocking) before snapping.
        if self.mStage is not None:
            try:
                self.mStage.move(value=float(center_x), axis="X", is_absolute=True, is_blocking=True)
                self.mStage.move(value=float(center_y), axis="Y", is_absolute=True, is_blocking=True)
            except Exception as e:
                self._logger.warning(f"recaptureSlot: stage move failed: {e}")

        # Re-use the existing snap logic so the snapshot is stored consistently.
        result = self.snapOverviewImage(slot_id=str(slot_id), camera_name=camera_name)
        result["recapturedSlot"] = slot_id
        result["slotCenter"] = {"x": float(center_x), "y": float(center_y)}
        return result

    @APIExport(requestType="POST")
    def registerOverviewSlide(self, registration_data: dict):
        """
        Save corner picks and compute per-slide registration (homography).

        Args:
            registration_data: dict with keys:
                - cameraName (str)
                - layoutName (str)
                - slotId (str)
                - slotName (str)
                - snapshotId (str)
                - snapshotTimestamp (str)
                - imageWidth (int)
                - imageHeight (int)
                - cornersPx: [{x, y}, ...] – 4 clicked corners in image pixels
                - slotStageCorners: [{x, y}, ...] – 4 target stage corners

        Returns:
            Registration metadata including homography and error metrics.
        """
        try:
            corners_px = [
                PixelPoint(x=c["x"], y=c["y"])
                for c in registration_data["cornersPx"]
            ]
            slot_stage_corners = [
                StagePoint(x=c["x"], y=c["y"])
                for c in registration_data["slotStageCorners"]
            ]

            # Try to load the raw snapshot for warping
            raw_image = None
            snapshot_id = registration_data.get("snapshotId", "")
            cam_name = registration_data.get("cameraName", self._overview_camera_name or "overviewcamera")
            layout_name = registration_data.get("layoutName", "Heidstar 4x Histosample")

            img_path = self._overview_registration.get_snapshot_image_path(
                cam_name, "current", snapshot_id
            )
            if img_path:
                import cv2
                raw_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            reg = self._overview_registration.register_slide(
                camera_name=cam_name,
                layout_name=layout_name,
                slot_id=registration_data["slotId"],
                slot_name=registration_data.get("slotName", f"Slide{registration_data['slotId']}"),
                snapshot_id=snapshot_id,
                snapshot_timestamp=registration_data.get("snapshotTimestamp", ""),
                image_width=registration_data.get("imageWidth", 0),
                image_height=registration_data.get("imageHeight", 0),
                corners_px=corners_px,
                slot_stage_corners=slot_stage_corners,
                raw_image=raw_image,
                stage_position=registration_data.get("stagePosition"),
            )

            return {
                "success": True,
                "slotId": reg.slotId,
                "reprojectionError": reg.reprojectionError,
                "hasOverlayImage": bool(reg.overlayImageRef),
                "cornerOrder": reg.cornerOrder,
                "createdAt": reg.createdAt,
            }
        except Exception as e:
            self._logger.error(f"Registration failed: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            try:
                self._persistOverviewRegistrationToSetup(
                    cam_name, layout_name
                )
            except Exception as _exc:
                self._logger.debug(
                    f"Skip persisting overview registration: {_exc}"
                )

    @APIExport(requestType="GET")
    def getOverviewRegistrationStatus(self, camera_name: str = "", layout_name: str = "Heidstar 4x Histosample"):
        """
        Get completion status for all slide registrations.

        Args:
            camera_name: Camera name (auto-detected if empty)
            layout_name: Layout name

        Returns:
            Per-slide completion status and metadata.
        """
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        return self._overview_registration.get_status(cam_name, layout_name)

    @APIExport(requestType="POST")
    def refreshOverviewSlideImage(self, slot_id: str = "1", camera_name: str = "", layout_name: str = "Heidstar 4x Histosample"):
        """
        Snap a new image and re-warp using existing registration for a slot.
        Does not require new corner picking if registration already exists.

        Args:
            slot_id:     Slide slot index ("1" to "4")
            camera_name: Camera name
            layout_name: Layout name

        Returns:
            Updated overlay metadata.
        """
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        cam = self._overview_camera
        if cam is None:
            raise HTTPException(status_code=400, detail="Overview camera not available")

        frame = cam.getLatestFrame()
        if frame is None:
            raise HTTPException(status_code=500, detail="No frame from overview camera")


        frame = np.ascontiguousarray(frame)

        try:
            result = self._overview_registration.refresh_overlay_image(
                camera_name=cam_name,
                layout_name=layout_name,
                slot_id=slot_id,
                new_image=frame,
            )
            return {"success": True, **result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @APIExport(requestType="GET")
    def getOverviewOverlayData(self, camera_name: str = "", layout_name: str = "Heidstar 4x Histosample"):
        """
        Get all overlay data for the WellSelector canvas rendering.
        Returns base64-encoded overlay images + slot bounds per completed slide.

        Args:
            camera_name: Camera name (auto-detected if empty)
            layout_name: Layout name

        Returns:
            Per-slide overlay images and stage bounds for canvas rendering.
        """
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        return self._overview_registration.get_overlay_data(cam_name, layout_name)

    # ------------------------------------------------------------------
    # Autonomous overview scan + registration config endpoints
    # ------------------------------------------------------------------
    # Async overview-task plumbing
    # ------------------------------------------------------------------
    def _tryStartOverviewAsync(self, task_name: str) -> bool:
        """Reserve the single async overview slot. Returns False if busy."""
        with self._overview_async_lock:
            t = self._overview_async_thread
            if t is not None and t.is_alive():
                return False
            self._overview_async_status = {
                "running": True,
                "task": task_name,
                "result": None,
                "error": None,
                "message": "",
            }
            self._overview_async_thread = None
            return True

    def _finishOverviewAsync(self, result=None, error=None) -> None:
        with self._overview_async_lock:
            self._overview_async_status = {
                "running": False,
                "task": self._overview_async_status.get("task"),
                "result": result,
                "error": error,
            }

    @APIExport(requestType="GET")
    def getOverviewAsyncStatus(self):
        """Return the current background-overview-task state for polling."""
        with self._overview_async_lock:
            return dict(self._overview_async_status)

    @APIExport(requestType="GET")
    def getKnownCalibrationPoint(self, layout_name: str = "Heidstar 4x Histosample"):
        """Return the expected (x_um, y_um) calibration centre for a layout.

        The frontend uses this as the "true" position when storing a stage
        offset: ``offset = expected - actual``.
        """
        ref = self._KNOWN_CALIBRATION_POINTS.get(layout_name)
        if ref is None:
            return {"success": False, "layoutName": layout_name, "known": None}
        return {"success": True, "layoutName": layout_name, "known": dict(ref)}

    @APIExport(requestType="GET")
    def getKnownCalibrationLayouts(self):
        """List every layout that ships a known calibration reference point."""
        return {
            "layouts": [
                {"name": name, **dict(data)}
                for name, data in self._KNOWN_CALIBRATION_POINTS.items()
            ]
        }

    # ------------------------------------------------------------------
    @APIExport(requestType="POST")
    def runAutonomousOverviewScan(
        self,
        camera_name: str = "",
        layout_name: str = "Heidstar 4x Histosample",
        settle_time_s: float = 0.5,
    ):
        """Schedule an autonomous overview scan and return immediately.

        The actual stage iteration + snap loop runs in a background thread so
        the FastAPI worker stays free. Poll :meth:`getOverviewAsyncStatus` for
        completion and the final ``scanResults`` / ``overlayData`` payload.
        """
        if not self._tryStartOverviewAsync("autonomous_overview_scan"):
            return {
                "started": False,
                "error": "Another overview task is still running",
                "status": self._overview_async_status,
            }

        thread = threading.Thread(
            target=self._runAutonomousOverviewScanWorker,
            args=(camera_name, layout_name, settle_time_s),
            daemon=True,
            name="runAutonomousOverviewScan",
        )
        with self._overview_async_lock:
            self._overview_async_thread = thread
        thread.start()
        return {"started": True, "task": "autonomous_overview_scan"}

    def _runAutonomousOverviewScanWorker(
        self,
        camera_name: str,
        layout_name: str,
        settle_time_s: float,
    ) -> None:
        try:
            result = self._runAutonomousOverviewScanBlocking(
                camera_name=camera_name,
                layout_name=layout_name,
                settle_time_s=settle_time_s,
            )
            self._finishOverviewAsync(result=result)
        except Exception as exc:
            self._logger.error(
                f"runAutonomousOverviewScan worker failed: {exc}", exc_info=True
            )
            self._finishOverviewAsync(error=str(exc))

    def _runAutonomousOverviewScanBlocking(
        self,
        camera_name: str = "",
        layout_name: str = "Heidstar 4x Histosample",
        settle_time_s: float = 0.5,
    ) -> dict:
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        cam = self._overview_camera
        if cam is None:
            raise HTTPException(status_code=400, detail="Overview camera not available")

        config = self._overview_registration.load_registration_config(cam_name, layout_name)
        if not config or not config.get("slots"):
            raise HTTPException(
                status_code=400,
                detail="No registration config found. Run the manual wizard first.",
            )

        results: Dict[str, Any] = {}
        for slot_id, slot_data in config["slots"].items():
            pos = slot_data.get("stagePosition") or {}
            try:
                # Move XY together when possible, then Z separately
                if self.mStage is not None:
                    if "x" in pos and "y" in pos:
                        self.mStage.move(
                            value=(float(pos["x"]), float(pos["y"])),
                            axis="XY",
                            is_absolute=True,
                            is_blocking=True,
                        )
                    if "z" in pos:
                        self.mStage.move(
                            value=float(pos["z"]),
                            axis="Z",
                            is_absolute=True,
                            is_blocking=True,
                        )
                if settle_time_s and settle_time_s > 0:
                    time.sleep(float(settle_time_s))

                frame = cam.getLatestFrame()
                if frame is None:
                    results[slot_id] = {"success": False, "error": "No frame from camera"}
                    continue
                frame = np.ascontiguousarray(frame)

                refresh = self._overview_registration.refresh_overlay_image(
                    camera_name=cam_name,
                    layout_name=layout_name,
                    slot_id=slot_id,
                    new_image=frame,
                )
                results[slot_id] = {"success": True, **refresh}
            except Exception as exc:
                self._logger.error(
                    f"Autonomous scan failed for slot {slot_id}: {exc}", exc_info=True
                )
                results[slot_id] = {"success": False, "error": str(exc)}

        overlay_data = self._overview_registration.get_overlay_data(cam_name, layout_name)
        return {"success": True, "scanResults": results, "overlayData": overlay_data}

    @APIExport(requestType="GET")
    def getOverviewRegistrationConfigData(
        self,
        camera_name: str = "",
        layout_name: str = "Heidstar 4x Histosample",
    ):
        """Return the persisted overview-registration config for editing."""
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        config = self._overview_registration.load_registration_config(cam_name, layout_name)
        if config is None:
            return {"exists": False, "config": None}
        return {"exists": True, "config": config}

    @APIExport(requestType="POST")
    def updateOverviewRegistrationConfig(self, config_data: dict):
        """Apply edits to the registration config (e.g. updated XYZ positions)."""
        if not isinstance(config_data, dict):
            raise HTTPException(status_code=400, detail="config_data must be an object")
        cam_name = config_data.get(
            "cameraName", self._overview_camera_name or "overviewcamera"
        )
        layout_name = config_data.get("layoutName", "Heidstar 4x Histosample")
        try:
            path = self._overview_registration.save_registration_config_from_dict(
                cam_name, layout_name, config_data
            )
            self._persistOverviewRegistrationToSetup(cam_name, layout_name)
            return {"success": True, "path": path}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @APIExport(requestType="GET")
    def getOverviewOverlayImage(
        self,
        slot_id: str = "1",
        camera_name: str = "",
        layout_name: str = "Heidstar 4x Histosample",
    ):
        """Return the stored JPEG overlay (base64) for a single slot."""
        cam_name = camera_name or self._overview_camera_name or "overviewcamera"
        b64 = self._overview_registration.load_overlay_jpeg(
            cam_name, layout_name, slot_id
        )
        if b64 is None:
            raise HTTPException(
                status_code=404, detail="No overlay image found for this slot"
            )
        return {
            "imageBase64": b64,
            "imageMimeType": "image/jpeg",
            "slotId": slot_id,
        }

    # ------------------------------------------------------------------
    # Overview registration <-> setup-config persistence helpers
    # ------------------------------------------------------------------
    def _overviewRegistrationKey(self, camera_name: str, layout_name: str) -> str:
        """Composite key used inside ``_setupInfo.overviewRegistration``."""
        return f"{camera_name}__{layout_name}"

    def _loadOverviewRegistrationFromSetup(self):
        """Push any persisted entries from the setup file into the in-memory
        registration store, so they survive process restarts."""
        store = getattr(self._setupInfo, "overviewRegistration", None)
        if not store or not isinstance(store, dict):
            return
        for key, cfg in store.items():
            if not isinstance(cfg, dict):
                continue
            cam_name = cfg.get("cameraName") or self._overview_camera_name or "overviewcamera"
            layout_name = cfg.get("layoutName") or "Heidstar 4x Histosample"
            try:
                self._overview_registration.save_registration_config_from_dict(
                    cam_name, layout_name, cfg
                )
            except Exception as exc:
                self._logger.warning(
                    f"Failed to load overview registration '{key}': {exc}"
                )

    def _persistOverviewRegistrationToSetup(self, camera_name: str, layout_name: str):
        """Mirror the latest registration config into the imcontrol_setups
        JSON (single source of truth, alongside ``stageOffsets``)."""
        cfg = self._overview_registration.load_registration_config(
            camera_name, layout_name
        )
        if cfg is None:
            return
        if getattr(self._setupInfo, "overviewRegistration", None) is None:
            self._setupInfo.overviewRegistration = {}
        key = self._overviewRegistrationKey(camera_name, layout_name)
        self._setupInfo.overviewRegistration[key] = cfg
        try:
            options = configfiletools.loadOptions()[0]
            configfiletools.saveSetupInfo(options, self._setupInfo)
        except Exception as exc:
            self._logger.warning(
                f"saveSetupInfo failed for overview registration: {exc}"
            )


# Copyright (C) 2025 Benedict Diederich

    def _init_overview(self):
        """Overview camera + registration store; async bookkeeping for the overview endpoints."""
        # Initialize overview camera registration service
        self._overview_registration = OverviewRegistrationService()
        # Background-task bookkeeping for overview-related endpoints. These
        # endpoints (recaptureSlot, runAutonomousOverviewScan, ...) used to
        # block the FastAPI worker thread for many seconds, freezing the UI.
        # They now spawn worker threads and clients poll
        # ``getOverviewAsyncStatus`` for completion.
        self._overview_async_lock = threading.Lock()
        self._overview_async_thread: Optional[threading.Thread] = None
        self._overview_async_status: dict = {
            "running": False,
            "task": None,
            "result": None,
            "error": None,
        }
        self._ashlar_proc = None  # subprocess.Popen handle for the running ashlar process
        # Overview camera is resolved through the DetectorManager, using the
        # detector name configured in setup (experiment.overviewCameraName).
        # Falls back to a detector literally named "overviewcamera" if any.
        self._overview_camera = None
        self._overview_camera_name = None
        try:
            preferred = getattr(self._master.experimentManager, "overviewCameraName", None)
            allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
            candidates = []
            if preferred:
                candidates.append(preferred)
            candidates.append("ObservationCamera")
            for name in candidates:
                if name and name in allDetectorNames:
                    self._overview_camera_name = name
                    self._overview_camera = self._master.detectorsManager[name]
                    self._logger.info(f"Overview camera resolved to detector '{name}'")
                    break
            if self._overview_camera is None:
                self._logger.info(
                    "No overview camera configured; overview features will return errors"
                )
        except Exception as exc:
            self._logger.warning(f"Failed to resolve overview camera: {exc}")

        # Load persisted overview registration entries from the setup config
        # (single source of truth) into the in-memory store.
        try:
            self._loadOverviewRegistrationFromSetup()
        except Exception as exc:
            self._logger.warning(
                f"Could not load overview registration from setup file: {exc}"
            )
