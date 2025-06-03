import os
import time
import threading
import collections
import tifffile as tif
from pathlib import Path
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController


class StageScanAcquisitionController(ImConWidgetController):
    """Couples a 2‑D stage scan with external‑trigger camera acquisition.

    • Puts the connected ``CameraHIK`` into *external* trigger mode
      (one exposure per TTL rising edge on LINE0).
    • Runs ``positioner.start_stage_scanning``.
    • Pops every frame straight from the camera ring‑buffer and writes it to
      disk as ``000123.tif`` (frame‑id used as filename).

    Assumes the micro‑controller (or the positioner itself) raises a TTL pulse
    **after** arriving at each grid co‑ordinate.
    """

    def __init__(self, *args, save_dir: str | os.PathLike | None = None, **kwargs):
        super().__init__(*args, **kwargs)
