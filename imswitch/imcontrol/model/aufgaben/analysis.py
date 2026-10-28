"""Analysis steps: measure what the run has seen and feed it back into the job.

Each step reads an image the run already has: the last frame of a capture step
or a derived image such as a prescan overview. ``source`` names that step; empty
takes the newest image. The result is plain data that expressions read as
``result.<step id>``, e.g. ``result.cells.count`` in a condition or
``${result.find.positions}`` as the source of a position loop.

Pixel positions become stage µm through the frame's pixel size and the stage
position it was taken at (``Frame.to_stage``). Frames arrive in stage
orientation (the camera applies the calibration's flips); rotation is not
corrected.

The script steps (``python_script``, ``external_process``) run unsandboxed code;
they need the ``scripting`` capability, which a setup grants explicitly, and a
run that contains them must be confirmed.
"""
from __future__ import annotations

import subprocess
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from annotated_types import Ge, Gt, Le
from pydantic import Field

from .hardware import Microscope
from .registry import TaskContext, TaskFlags, task

MAX_LISTED = 500          # objects listed in a result; counts are always complete
OUTPUT_CHARS = 4000       # tail of a process' output kept in the result

Fraction = Annotated[float, Gt(0), Le(1)]


# --------------------------------------------------------------------------- #
# Image helpers (pure; tested on synthetic frames)
# --------------------------------------------------------------------------- #
def gray(data: Any) -> np.ndarray:
    """A 2-D float image: RGB averaged, singleton axes dropped."""
    a = np.squeeze(np.asarray(data))
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a[..., :3].mean(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {a.shape}")
    return a.astype(np.float32, copy=False)


def center_crop(a: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 1:
        return a
    h, w = a.shape[:2]
    ch, cw = max(1, round(h * fraction)), max(1, round(w * fraction))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return a[y0:y0 + ch, x0:x0 + cw]


def intensity_stats(data: Any, center_fraction: float = 1.0) -> Dict[str, float]:
    raw = center_crop(np.asarray(data), center_fraction)
    a = gray(raw)
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    top = np.iinfo(raw.dtype).max if raw.dtype.kind in "ui" else None
    saturated = raw >= top if top is not None else np.zeros(raw.shape, bool)
    if saturated.ndim == 3:
        saturated = saturated.any(axis=-1)
    return {"mean": float(a.mean()), "median": float(p50), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "p1": float(p1), "p99": float(p99),
            "saturated_fraction": float(saturated.mean())}


def focus_value(data: Any, method: str = "laplacian", center_fraction: float = 1.0) -> float:
    """Sharpness; larger is sharper. Compare values from the same exposure only."""
    from scipy import ndimage as ndi
    a = center_crop(gray(data), center_fraction)
    if method == "laplacian":
        return float(ndi.laplace(a).var())
    if method == "tenengrad":
        return float(np.mean(ndi.sobel(a, 0) ** 2 + ndi.sobel(a, 1) ** 2))
    if method == "normalized_variance":
        mean = float(a.mean())
        return float(a.var() / mean) if mean > 0 else 0.0
    raise ValueError(f"unknown focus method {method!r}")


def find_objects(a: np.ndarray, threshold: Optional[float] = None, dark: bool = False,
                 smooth_px: float = 1.0, min_area_px: int = 1,
                 max_area_px: Optional[int] = None, grow_px: int = 0
                 ) -> Tuple[float, List[Dict[str, Any]]]:
    """Threshold (Otsu when ``threshold`` is None) and label connected objects.

    ``grow_px`` dilates the mask first, so objects closer than that merge.
    Returns the threshold and per object: area_px, centroid (row, col) and
    bounding box (row0, col0, row1, col1), largest first.
    """
    from scipy import ndimage as ndi
    img = ndi.gaussian_filter(a, smooth_px) if smooth_px > 0 else a
    if threshold is None:
        if float(img.max()) <= float(img.min()):
            return float(img.max()), []
        from skimage.filters import threshold_otsu
        threshold = float(threshold_otsu(img))
    mask = img < threshold if dark else img > threshold
    if grow_px > 0:
        mask = ndi.binary_dilation(mask, iterations=int(grow_px))
    labels, n = ndi.label(mask)
    if n == 0:
        return float(threshold), []
    index = np.arange(1, n + 1)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    centroids = ndi.center_of_mass(mask, labels, index)
    objects = []
    for i, (area, centroid, box) in enumerate(zip(areas, centroids, ndi.find_objects(labels))):
        if area < min_area_px or (max_area_px is not None and area > max_area_px):
            continue
        objects.append({"area_px": int(area), "centroid": (float(centroid[0]), float(centroid[1])),
                        "box": (box[0].start, box[1].start, box[0].stop, box[1].stop)})
    objects.sort(key=lambda o: -o["area_px"])
    return float(threshold), objects


def _detect(ctx: TaskContext, source, threshold, dark, smooth_um, min_area_um2,
            max_area_um2=None, grow_um=0.0):
    frame = ctx.frame(source)
    px = frame.pixel_size_um
    threshold, objects = find_objects(
        gray(frame.data), threshold=threshold, dark=dark, smooth_px=smooth_um / px,
        min_area_px=max(1, int(min_area_um2 / px ** 2)),
        max_area_px=None if max_area_um2 is None else int(max_area_um2 / px ** 2),
        grow_px=int(round(grow_um / px)))
    return frame, threshold, objects


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
SOURCE_DOC = "Step whose last image to read (a capture or prescan); empty takes the newest."
THRESHOLD_DOC = "Intensity threshold; empty finds one automatically (Otsu)."


@task(category="Analysis", effects="NONE")
def measure_intensity(source: Optional[str] = None, center_fraction: Fraction = 1.0,
                      *, ctx: TaskContext) -> Dict[str, float]:
    """Measure intensity

    Mean, median, spread, percentiles and the saturated fraction of an image,
    e.g. to adjust the exposure or to skip a dark field.

    Args:
        source: Step whose last image to read (a capture or prescan); empty takes the newest.
        center_fraction: Measure only this central part of the image (1 = all).
    """
    return intensity_stats(ctx.frame(source).data, center_fraction)


@task(category="Analysis", effects="NONE")
def focus_metric(source: Optional[str] = None,
                 method: Literal["laplacian", "tenengrad", "normalized_variance"] = "laplacian",
                 center_fraction: Fraction = 1.0, *, ctx: TaskContext) -> Dict[str, Any]:
    """Focus metric

    How sharp an image is (larger is sharper), e.g. to repeat the autofocus
    when the value drops. Compare values taken with the same exposure.

    Args:
        source: Step whose last image to read (a capture or prescan); empty takes the newest.
        method: Laplacian variance, Tenengrad gradient energy or normalised variance.
        center_fraction: Measure only this central part of the image (1 = all).
    """
    return {"value": focus_value(ctx.frame(source).data, method, center_fraction),
            "method": method}


@task(category="Analysis", effects="NONE")
def detect_background(source: Optional[str] = None,
                      max_contrast: Annotated[float, Gt(0)] = 0.05,
                      smooth_um: Annotated[float, Ge(0)] = 2.0,
                      center_fraction: Fraction = 1.0, *, ctx: TaskContext) -> Dict[str, Any]:
    """Detect background

    Says whether a field is empty: its contrast (standard deviation over mean
    after smoothing) stays below ``max_contrast``. Use it to skip empty wells
    or positions.

    Args:
        source: Step whose last image to read (a capture or prescan); empty takes the newest.
        max_contrast: Contrast below which the field counts as empty.
        smooth_um: Smoothing before measuring, to ignore camera noise.
        center_fraction: Judge only this central part of the image (1 = all).
    """
    from scipy import ndimage as ndi
    frame = ctx.frame(source)
    a = center_crop(gray(frame.data), center_fraction)
    if smooth_um > 0:
        a = ndi.gaussian_filter(a, smooth_um / frame.pixel_size_um)
    mean = float(a.mean())
    contrast = float(a.std() / mean) if mean > 0 else 0.0
    return {"empty": contrast < max_contrast, "contrast": contrast, "mean": mean}


@task(category="Analysis", effects="NONE")
def count_objects(source: Optional[str] = None, threshold: Optional[float] = None,
                  dark_objects: bool = False, min_area_um2: Annotated[float, Ge(0)] = 10.0,
                  max_area_um2: Optional[Annotated[float, Gt(0)]] = None,
                  smooth_um: Annotated[float, Ge(0)] = 1.0,
                  *, ctx: TaskContext) -> Dict[str, Any]:
    """Count objects

    Thresholds an image and counts the connected objects in a size range
    (cells, beads, colonies), with their stage positions and areas.

    Args:
        source: Step whose last image to read (a capture or prescan); empty takes the newest.
        threshold: Intensity threshold; empty finds one automatically (Otsu).
        dark_objects: Objects darker than the background (e.g. brightfield).
        min_area_um2: Smaller objects are ignored.
        max_area_um2: Larger objects are ignored; empty keeps all.
        smooth_um: Smoothing before thresholding.
    """
    frame, threshold, objects = _detect(ctx, source, threshold, dark_objects, smooth_um,
                                        min_area_um2, max_area_um2)
    area_um2 = frame.pixel_size_um ** 2
    listed = []
    for obj in objects[:MAX_LISTED]:
        x, y = frame.to_stage(*obj["centroid"])
        listed.append({"x": round(x, 2), "y": round(y, 2),
                       "area_um2": round(obj["area_px"] * area_um2, 2)})
    total = sum(o["area_px"] for o in objects)
    return {"count": len(objects), "threshold": threshold, "objects": listed,
            "mean_area_um2": total * area_um2 / len(objects) if objects else 0.0,
            "area_fraction": total / gray(frame.data).size}


@task(category="Analysis", effects="NONE")
def points_from_mask(source: Optional[str] = None, threshold: Optional[float] = None,
                     dark_objects: bool = False, min_area_um2: Annotated[float, Ge(0)] = 10.0,
                     max_area_um2: Optional[Annotated[float, Gt(0)]] = None,
                     smooth_um: Annotated[float, Ge(0)] = 1.0,
                     max_points: Annotated[int, Ge(1)] = 50, name_prefix: str = "obj",
                     *, ctx: TaskContext) -> Dict[str, Any]:
    """Points from mask

    Finds objects like count_objects and returns their centres as stage
    positions, largest first, for a position loop:
    ``"source": "${result.<this step>.positions}"``.

    Args:
        source: Step whose last image to read (a capture or prescan); empty takes the newest.
        threshold: Intensity threshold; empty finds one automatically (Otsu).
        dark_objects: Objects darker than the background (e.g. brightfield).
        min_area_um2: Smaller objects are ignored.
        max_area_um2: Larger objects are ignored; empty keeps all.
        smooth_um: Smoothing before thresholding.
        max_points: At most this many positions, the largest objects.
        name_prefix: Positions are named prefix0, prefix1, ...
    """
    frame, _, objects = _detect(ctx, source, threshold, dark_objects, smooth_um,
                                min_area_um2, max_area_um2)
    positions = []
    for i, obj in enumerate(objects[:max_points]):
        x, y = frame.to_stage(*obj["centroid"])
        positions.append({"x": round(x, 2), "y": round(y, 2), "name": f"{name_prefix}{i}"})
    return {"positions": positions, "count": len(positions), "found": len(objects)}


@task(category="Analysis", effects="NONE")
def regions_from_overview(source: Optional[str] = None, threshold: Optional[float] = None,
                          dark_objects: bool = False,
                          min_area_um2: Annotated[float, Ge(0)] = 1000.0,
                          margin_um: Annotated[float, Ge(0)] = 50.0,
                          smooth_um: Annotated[float, Ge(0)] = 5.0,
                          max_regions: Annotated[int, Ge(1)] = 20, name_prefix: str = "region",
                          *, ctx: TaskContext) -> Dict[str, Any]:
    """Regions from overview

    Finds the sample in an overview (e.g. a prescan) and returns one
    rectangle per piece, grown by a margin, for a region loop:
    ``"regions": "${result.<this step>.regions}"``. Pieces closer than the
    margin become one region.

    Args:
        source: Step whose last image to read (a capture or prescan); empty takes the newest.
        threshold: Intensity threshold; empty finds one automatically (Otsu).
        dark_objects: Sample darker than the background (e.g. brightfield).
        min_area_um2: Smaller pieces are ignored.
        margin_um: Added around each piece; pieces this close merge.
        smooth_um: Smoothing before thresholding.
        max_regions: At most this many regions, the largest.
        name_prefix: Regions are named prefix0, prefix1, ...
    """
    frame, _, objects = _detect(ctx, source, threshold, dark_objects, smooth_um,
                                min_area_um2, grow_um=margin_um)
    regions = []
    for i, obj in enumerate(objects[:max_regions]):
        r0, c0, r1, c1 = obj["box"]
        x0, y0 = frame.to_stage(r0, c0)
        x1, y1 = frame.to_stage(r1, c1)
        regions.append({"name": f"{name_prefix}{i}", "min_x": round(x0, 2), "max_x": round(x1, 2),
                        "min_y": round(y0, 2), "max_y": round(y1, 2)})
    return {"regions": regions, "count": len(regions)}


@task(category="Scripts", effects="UNKNOWN",
      flags=TaskFlags(runs_code=True, needs=("scripting",)))
def python_script(code: Annotated[str, Field(json_schema_extra={"x-widget": "code"})],
                  *, hw: Microscope, ctx: TaskContext) -> Dict[str, Any]:
    """Python script

    Runs Python code in the engine, unsandboxed: it can drive the hardware.
    Names available: ``np``, ``hw`` (the microscope), ``frame(source=None)``
    (an image the run has seen), ``var`` (the job's variables, writable),
    ``result`` (what the steps returned) and ``log(text)``. Whatever the code
    assigns to ``output`` is this step's ``result.<id>.output``.

    Args:
        code: Python source code.
    """
    namespace = {"np": np, "hw": hw, "frame": ctx.frame, "var": ctx.variables,
                 "result": ctx.results, "log": ctx.log, "output": None}
    exec(compile(code, "<python_script>", "exec"), namespace)  # noqa: S102 (ask-first, opt-in)
    return {"output": namespace["output"]}


@task(category="Scripts", effects="UNKNOWN",
      flags=TaskFlags(runs_code=True, external=True, needs=("scripting",)))
def external_process(command: List[str], timeout_s: Annotated[float, Gt(0)] = 60.0,
                     check: bool = True) -> Dict[str, Any]:
    """External process

    Runs a program (no shell) and returns its exit code and output, e.g. an
    analysis script that writes a result file.

    Args:
        command: Program and its arguments, one item each.
        timeout_s: Stop the program after this long.
        check: Fail the step when the program exits with an error.
    """
    if not command:
        raise ValueError("external_process: empty command")
    done = subprocess.run(command, capture_output=True, text=True,  # noqa: S603 (ask-first)
                          timeout=timeout_s)
    if check and done.returncode != 0:
        raise RuntimeError(f"{command[0]} exited with {done.returncode}: "
                           f"{done.stderr[-OUTPUT_CHARS:].strip()}")
    return {"exit_code": done.returncode, "stdout": done.stdout[-OUTPUT_CHARS:],
            "stderr": done.stderr[-OUTPUT_CHARS:]}
