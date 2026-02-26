"""
Overview Registration Service Module
=====================================

Provides per-slide registration (homography-based) for oblique overview camera images,
mapping overview camera pixel coordinates to stage coordinates.

Designed for the 4-slide Heidstar adapter, but extensible to other layouts.

Copyright (C) 2025 Benedict Diederich
"""

import json
import os
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from imswitch.imcommon.model import dirtools, initLogger

import logging
logger = logging.getLogger("OverviewRegistration")


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PixelPoint(BaseModel):
    """A point in image pixel coordinates."""
    x: float
    y: float


class StagePoint(BaseModel):
    """A point in stage / layout coordinates (µm)."""
    x: float
    y: float


class SlotDefinition(BaseModel):
    """A slide slot rectangle in stage coordinates."""
    slotId: str
    name: str
    x: float           # center x in µm
    y: float           # center y in µm
    width: float        # width in µm
    height: float       # height in µm
    shape: str = "rectangle"
    corners: List[StagePoint] = Field(default_factory=list)
    # corners in order: TL, TR, BR, BL (stage coordinates)


class SlideRegistration(BaseModel):
    """Stored per-slide registration data."""
    cameraName: str
    layoutName: str
    slotId: str
    slotName: str
    snapshotId: str
    snapshotTimestamp: str
    imageWidth: int
    imageHeight: int
    cornersPx: List[PixelPoint]         # 4 user-clicked corners in image px
    slotStageCorners: List[StagePoint]  # 4 target corners in stage coords
    cornerOrder: str = "TL,TR,BR,BL"   # convention
    homography: List[List[float]]       # 3x3 matrix as nested list
    inverseHomography: List[List[float]]  # 3x3 inverse for stage→px
    reprojectionError: float = 0.0
    overlayImageRef: str = ""           # relative path to warped overlay image
    createdAt: str = ""
    updatedAt: str = ""


class OverviewRegistrationStore(BaseModel):
    """Top-level store: all registrations for a layout+camera combo."""
    cameraName: str
    layoutName: str
    slides: Dict[str, SlideRegistration] = Field(default_factory=dict)
    # key = slotId ("1", "2", "3", "4")


# ── Helper Functions ─────────────────────────────────────────────────────────

def _slot_corners_from_well(well: dict) -> List[StagePoint]:
    """
    Compute the 4 corners (TL, TR, BR, BL) of a rectangular well
    from center-based well definition {x, y, width, height}.
    """
    cx = well["x"]
    cy = well["y"]
    w = well["width"]
    h = well["height"]
    half_w = w / 2.0
    half_h = h / 2.0
    return [
        StagePoint(x=cx - half_w, y=cy - half_h),  # TL
        StagePoint(x=cx + half_w, y=cy - half_h),  # TR
        StagePoint(x=cx + half_w, y=cy + half_h),  # BR
        StagePoint(x=cx - half_w, y=cy + half_h),  # BL
    ]


def compute_homography(
    src_points: List[PixelPoint],
    dst_points: List[StagePoint],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a projective (homography) transform from source pixel coords
    to destination stage coords.

    Args:
        src_points: 4 clicked pixel corners [TL, TR, BR, BL]
        dst_points: 4 target stage corners  [TL, TR, BR, BL]

    Returns:
        (H, H_inv, reprojection_error)
        H maps pixel → stage, H_inv maps stage → pixel.
    """
    src = np.array([[p.x, p.y] for p in src_points], dtype=np.float64)
    dst = np.array([[p.x, p.y] for p in dst_points], dtype=np.float64)

    H, status = cv2.findHomography(src, dst, method=0)  # exact 4-point, no RANSAC
    if H is None:
        raise ValueError("Homography computation failed – points may be degenerate.")

    H_inv = np.linalg.inv(H)

    # Reprojection error
    src_h = np.hstack([src, np.ones((4, 1))])
    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))

    return H, H_inv, error


def warp_slide_image(
    image: np.ndarray,
    H: np.ndarray,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """
    Warp the overview snapshot so the slide aligns with the target slot rectangle.
    The output is a cropped+warped image fitting the slot bounding box.

    Args:
        image:         raw overview snapshot (BGR or grayscale)
        H:             3x3 homography (pixel → stage)
        output_width:  desired output width in pixels
        output_height: desired output height in pixels

    Returns:
        Warped image (RGBA with transparency outside the slide polygon)
    """
    warped = cv2.warpPerspective(
        image, H, (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


# ── Main Service Class ───────────────────────────────────────────────────────

class OverviewRegistrationService:
    """
    Manages overview camera registration for slide slots.
    Persists data to JSON + image files on disk.
    """

    def __init__(self, save_base_dir: Optional[str] = None):
        self._logger = initLogger(self, instanceName="OverviewRegistrationService")
        if save_base_dir is None:
            save_base_dir = dirtools.UserFileDirs.getValidatedDataPath()
        self._base_dir = os.path.join(save_base_dir, "OverviewRegistration")
        os.makedirs(self._base_dir, exist_ok=True)
        self._stores: Dict[str, OverviewRegistrationStore] = {}
        self._load_all()

    # ── Persistence ──────────────────────────────────────────────────────

    def _store_key(self, camera_name: str, layout_name: str) -> str:
        return f"{camera_name}__{layout_name}".replace(" ", "_")

    def _store_dir(self, camera_name: str, layout_name: str) -> str:
        key = self._store_key(camera_name, layout_name)
        d = os.path.join(self._base_dir, key)
        os.makedirs(d, exist_ok=True)
        return d

    def _meta_path(self, camera_name: str, layout_name: str) -> str:
        return os.path.join(
            self._store_dir(camera_name, layout_name),
            "registrations.json",
        )

    def _load_all(self):
        """Load all persisted registration stores from disk."""
        if not os.path.isdir(self._base_dir):
            return
        for entry in os.listdir(self._base_dir):
            meta_file = os.path.join(self._base_dir, entry, "registrations.json")
            if os.path.isfile(meta_file):
                try:
                    with open(meta_file, "r") as f:
                        data = json.load(f)
                    store = OverviewRegistrationStore(**data)
                    key = self._store_key(store.cameraName, store.layoutName)
                    self._stores[key] = store
                    self._logger.info(f"Loaded registration store: {key}")
                except Exception as e:
                    self._logger.warning(f"Failed to load {meta_file}: {e}")

    def _save_store(self, camera_name: str, layout_name: str):
        key = self._store_key(camera_name, layout_name)
        store = self._stores.get(key)
        if store is None:
            return
        path = self._meta_path(camera_name, layout_name)
        with open(path, "w") as f:
            f.write(store.json(indent=2))
        self._logger.info(f"Saved registration store to {path}")

    # ── Slot Definitions ─────────────────────────────────────────────────

    def get_slot_definitions(self, layout: dict) -> List[SlotDefinition]:
        """
        Extract slide slot definitions from a well layout dict.
        Only rectangular wells are treated as slide slots.
        """
        slots = []
        idx = 1
        for well in layout.get("wells", []):
            if well.get("shape") == "rectangle":
                corners = _slot_corners_from_well(well)
                slots.append(SlotDefinition(
                    slotId=str(idx),
                    name=well.get("name", f"Slide{idx}"),
                    x=well["x"],
                    y=well["y"],
                    width=well["width"],
                    height=well["height"],
                    shape="rectangle",
                    corners=corners,
                ))
                idx += 1
        return slots

    # ── Snapshot ─────────────────────────────────────────────────────────

    def save_snapshot(
        self,
        camera_name: str,
        layout_name: str,
        slot_id: str,
        image: np.ndarray,
        stage_x: float = 0.0,
        stage_y: float = 0.0,
        stage_z: float = 0.0,
    ) -> dict:
        """
        Save a raw snapshot for a slot. Returns metadata dict.
        """
        sdir = self._store_dir(camera_name, layout_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"slot{slot_id}_{ts}"
        fname = f"{snapshot_id}.png"
        fpath = os.path.join(sdir, fname)

        # Ensure image is saveable
        if len(image.shape) == 2:
            cv2.imwrite(fpath, image)
        else:
            cv2.imwrite(fpath, image)

        h, w = image.shape[:2]

        return {
            "snapshotId": snapshot_id,
            "imagePath": fpath,
            "imageRelPath": fname,
            "timestamp": ts,
            "imageWidth": w,
            "imageHeight": h,
            "stageX": stage_x,
            "stageY": stage_y,
            "stageZ": stage_z,
        }

    # ── Registration ─────────────────────────────────────────────────────

    def register_slide(
        self,
        camera_name: str,
        layout_name: str,
        slot_id: str,
        slot_name: str,
        snapshot_id: str,
        snapshot_timestamp: str,
        image_width: int,
        image_height: int,
        corners_px: List[PixelPoint],
        slot_stage_corners: List[StagePoint],
        raw_image: Optional[np.ndarray] = None,
    ) -> SlideRegistration:
        """
        Compute homography and store per-slide registration.
        Optionally warps the raw image to create an overlay image.
        """
        if len(corners_px) != 4:
            raise ValueError("Exactly 4 corner points required.")
        if len(slot_stage_corners) != 4:
            raise ValueError("Exactly 4 slot stage corners required.")

        H, H_inv, error = compute_homography(corners_px, slot_stage_corners)

        # Create overlay image if raw image provided
        overlay_ref = ""
        if raw_image is not None:
            sdir = self._store_dir(camera_name, layout_name)
            overlay_fname = f"overlay_slot{slot_id}.png"
            overlay_path = os.path.join(sdir, overlay_fname)

            # Determine output size: map stage slot to a reasonable pixel resolution
            # Use ~2 px per µm as a reasonable default
            stage_w = abs(slot_stage_corners[1].x - slot_stage_corners[0].x)
            stage_h = abs(slot_stage_corners[2].y - slot_stage_corners[1].y)
            px_per_um = 0.5  # 0.5 px per µm → 500px per mm
            out_w = max(100, int(stage_w * px_per_um))
            out_h = max(100, int(stage_h * px_per_um))

            # We need a homography that maps image pixels → overlay pixels
            # overlay pixel (0,0) corresponds to slot TL in stage coords
            # overlay pixel (out_w, out_h) corresponds to slot BR
            tl = slot_stage_corners[0]
            br = slot_stage_corners[2]
            # Build stage→overlay_pixel transform
            sx = out_w / (br.x - tl.x) if (br.x - tl.x) != 0 else 1
            sy = out_h / (br.y - tl.y) if (br.y - tl.y) != 0 else 1
            # stage_to_overlay: first translate by -TL, then scale
            M_stage_to_overlay = np.array([
                [sx, 0, -tl.x * sx],
                [0, sy, -tl.y * sy],
                [0,  0,  1],
            ], dtype=np.float64)

            # Combined: image_px → stage → overlay_px
            H_combined = M_stage_to_overlay @ H

            # Add alpha channel if not present
            if len(raw_image.shape) == 2:
                raw_rgba = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGRA)
            elif raw_image.shape[2] == 3:
                raw_rgba = cv2.cvtColor(raw_image, cv2.COLOR_BGR2BGRA)
            else:
                raw_rgba = raw_image

            warped = cv2.warpPerspective(
                raw_rgba, H_combined, (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            cv2.imwrite(overlay_path, warped)
            overlay_ref = overlay_fname
            self._logger.info(
                f"Created overlay image {overlay_fname} ({out_w}x{out_h}), "
                f"reprojection error: {error:.2f} µm"
            )

        now = datetime.now().isoformat()
        reg = SlideRegistration(
            cameraName=camera_name,
            layoutName=layout_name,
            slotId=slot_id,
            slotName=slot_name,
            snapshotId=snapshot_id,
            snapshotTimestamp=snapshot_timestamp,
            imageWidth=image_width,
            imageHeight=image_height,
            cornersPx=corners_px,
            slotStageCorners=slot_stage_corners,
            cornerOrder="TL,TR,BR,BL",
            homography=[list(row) for row in H.tolist()],
            inverseHomography=[list(row) for row in H_inv.tolist()],
            reprojectionError=error,
            overlayImageRef=overlay_ref,
            createdAt=now,
            updatedAt=now,
        )

        # Store
        key = self._store_key(camera_name, layout_name)
        if key not in self._stores:
            self._stores[key] = OverviewRegistrationStore(
                cameraName=camera_name,
                layoutName=layout_name,
            )
        self._stores[key].slides[slot_id] = reg
        self._save_store(camera_name, layout_name)

        return reg

    # ── Status / Query ───────────────────────────────────────────────────

    def get_status(self, camera_name: str, layout_name: str) -> dict:
        """Return completion status for all slides."""
        key = self._store_key(camera_name, layout_name)
        store = self._stores.get(key)
        if store is None:
            return {"cameraName": camera_name, "layoutName": layout_name, "slides": {}}
        result = {
            "cameraName": camera_name,
            "layoutName": layout_name,
            "slides": {},
        }
        for sid, reg in store.slides.items():
            result["slides"][sid] = {
                "slotId": sid,
                "slotName": reg.slotName,
                "complete": True,
                "snapshotId": reg.snapshotId,
                "snapshotTimestamp": reg.snapshotTimestamp,
                "reprojectionError": reg.reprojectionError,
                "hasOverlayImage": bool(reg.overlayImageRef),
                "updatedAt": reg.updatedAt,
            }
        return result

    def get_registration(self, camera_name: str, layout_name: str, slot_id: str) -> Optional[SlideRegistration]:
        """Get a specific slide registration."""
        key = self._store_key(camera_name, layout_name)
        store = self._stores.get(key)
        if store is None:
            return None
        return store.slides.get(slot_id)

    # ── Refresh Overlay ──────────────────────────────────────────────────

    def refresh_overlay_image(
        self,
        camera_name: str,
        layout_name: str,
        slot_id: str,
        new_image: np.ndarray,
    ) -> dict:
        """
        Re-warp a new snapshot using existing registration for a slot.
        Returns updated overlay metadata.
        """
        reg = self.get_registration(camera_name, layout_name, slot_id)
        if reg is None:
            raise ValueError(f"No existing registration for slot {slot_id}")

        H = np.array(reg.homography, dtype=np.float64)
        slot_stage_corners = reg.slotStageCorners

        sdir = self._store_dir(camera_name, layout_name)
        overlay_fname = f"overlay_slot{slot_id}.png"
        overlay_path = os.path.join(sdir, overlay_fname)

        # Compute output size
        tl = slot_stage_corners[0]
        br = slot_stage_corners[2]
        stage_w = abs(br.x - tl.x)
        stage_h = abs(br.y - tl.y)
        px_per_um = 0.5
        out_w = max(100, int(stage_w * px_per_um))
        out_h = max(100, int(stage_h * px_per_um))

        sx = out_w / (br.x - tl.x) if (br.x - tl.x) != 0 else 1
        sy = out_h / (br.y - tl.y) if (br.y - tl.y) != 0 else 1
        M_stage_to_overlay = np.array([
            [sx, 0, -tl.x * sx],
            [0, sy, -tl.y * sy],
            [0,  0,  1],
        ], dtype=np.float64)
        H_combined = M_stage_to_overlay @ H

        if len(new_image.shape) == 2:
            img_rgba = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGRA)
        elif new_image.shape[2] == 3:
            img_rgba = cv2.cvtColor(new_image, cv2.COLOR_BGR2BGRA)
        else:
            img_rgba = new_image

        warped = cv2.warpPerspective(
            img_rgba, H_combined, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        cv2.imwrite(overlay_path, warped)

        # Update timestamp
        reg.updatedAt = datetime.now().isoformat()
        reg.overlayImageRef = overlay_fname

        # Save new snapshot too
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_fname = f"slot{slot_id}_{ts}.png"
        cv2.imwrite(os.path.join(sdir, snap_fname), new_image)
        reg.snapshotId = f"slot{slot_id}_{ts}"
        reg.snapshotTimestamp = ts

        self._save_store(camera_name, layout_name)

        return {
            "slotId": slot_id,
            "overlayImageRef": overlay_fname,
            "snapshotId": reg.snapshotId,
            "updatedAt": reg.updatedAt,
        }

    # ── Overlay Data for Frontend ────────────────────────────────────────

    def get_overlay_data(self, camera_name: str, layout_name: str) -> dict:
        """
        Return all overlay data needed by the frontend WellSelector canvas.
        Each completed slide includes base64-encoded PNG overlay + slot bounds.
        """
        key = self._store_key(camera_name, layout_name)
        store = self._stores.get(key)
        result = {
            "cameraName": camera_name,
            "layoutName": layout_name,
            "slides": {},
        }
        if store is None:
            return result

        sdir = self._store_dir(camera_name, layout_name)

        for sid, reg in store.slides.items():
            if not reg.overlayImageRef:
                continue
            overlay_path = os.path.join(sdir, reg.overlayImageRef)
            if not os.path.isfile(overlay_path):
                continue

            # Read and base64-encode the overlay image
            with open(overlay_path, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("ascii")

            # Slot bounds in stage coordinates
            corners = reg.slotStageCorners
            min_x = min(c.x for c in corners)
            max_x = max(c.x for c in corners)
            min_y = min(c.y for c in corners)
            max_y = max(c.y for c in corners)

            result["slides"][sid] = {
                "slotId": sid,
                "slotName": reg.slotName,
                "imageBase64": b64,
                "imageMimeType": "image/png",
                "stageBounds": {
                    "minX": min_x,
                    "maxX": max_x,
                    "minY": min_y,
                    "maxY": max_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y,
                },
                "reprojectionError": reg.reprojectionError,
                "updatedAt": reg.updatedAt,
            }

        return result

    def get_snapshot_image_path(self, camera_name: str, layout_name: str, snapshot_id: str) -> Optional[str]:
        """Return the full path to a snapshot image."""
        sdir = self._store_dir(camera_name, layout_name)
        fpath = os.path.join(sdir, f"{snapshot_id}.png")
        if os.path.isfile(fpath):
            return fpath
        return None

    def get_overlay_image_path(self, camera_name: str, layout_name: str, slot_id: str) -> Optional[str]:
        """Return the full path to the overlay image for a slot."""
        reg = self.get_registration(camera_name, layout_name, slot_id)
        if reg is None or not reg.overlayImageRef:
            return None
        sdir = self._store_dir(camera_name, layout_name)
        fpath = os.path.join(sdir, reg.overlayImageRef)
        if os.path.isfile(fpath):
            return fpath
        return None
