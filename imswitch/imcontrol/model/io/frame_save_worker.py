"""Background worker for offloading frame writes out of the workflow thread.

``ExperimentController.save_frame_ome`` used to call
``ome_writer.write_frame(img, metadata)`` synchronously, blocking the
workflow thread for the duration of the TIFF/Zarr serialisation
(20–100 ms per frame at 9 MP; more if the image is large and disk is
slow). For fast acquisitions this is the dominant per-frame cost.

This worker decouples the write from the workflow:

  - Single dedicated thread (preserves per-writer ordering since all
    writes for the same writer go through this worker FIFO).
  - Bounded queue; on overflow the producer **blocks** rather than
    drops — experiment data integrity matters more than smooth
    latency. The maxsize bounds peak memory (each pending task holds
    a frame copy).
  - Optional ``on_success(result)`` callback for downstream signals
    (e.g. ``sigUpdateOMEZarrStore``). Runs on the worker thread, so
    keep it cheap.

Same shape as ``recording_service.BackgroundStorageWorker`` but
without the priority-queue overhead — experiment writes are FIFO.
"""

import queue
import threading
import time
from typing import Any, Callable, Optional

from imswitch.imcommon.model import initLogger


class FrameSaveWorker:
    """Single-thread bounded FIFO of frame-write tasks."""

    def __init__(self, name: str = "FrameSaveWorker", maxsize: int = 32):
        self._logger = initLogger(self, instanceName=name)
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._completed = 0
        self._failed = 0
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True,
        )
        self._thread.start()
        self._logger.info(f"started (maxsize={maxsize})")

    def submit(
        self,
        fn: Callable,
        *args,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        block: bool = True,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """Queue ``fn(*args, **kwargs)`` for the worker.

        On overflow with ``block=True`` (default), waits until a slot
        opens. With ``block=False`` returns False immediately if full.
        """
        try:
            self._queue.put(
                (fn, args, kwargs, on_success, on_error),
                block=block,
                timeout=timeout,
            )
            return True
        except queue.Full:
            return False

    def pending(self) -> int:
        return self._queue.qsize()

    def stats(self) -> dict:
        return {
            "pending": self._queue.qsize(),
            "completed": self._completed,
            "failed": self._failed,
        }

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until the queue drains or ``timeout`` expires."""
        t_end = time.time() + timeout
        while not self._queue.empty() and time.time() < t_end:
            time.sleep(0.05)
        return self._queue.empty()

    def stop(self, drain: bool = True, timeout: float = 30.0):
        if drain:
            self.flush(timeout)
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._logger.info(
            f"stopped (completed={self._completed}, failed={self._failed})"
        )

    # ------------------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            fn, args, kwargs, on_success, on_error = task
            try:
                result = fn(*args, **kwargs)
                self._completed += 1
                if on_success is not None:
                    try:
                        on_success(result)
                    except Exception as cb_err:
                        self._logger.warning(
                            f"on_success callback raised: {cb_err}"
                        )
            except Exception as e:
                self._failed += 1
                self._logger.error(f"task failed: {e}")
                if on_error is not None:
                    try:
                        on_error(e)
                    except Exception:
                        pass
            finally:
                self._queue.task_done()
