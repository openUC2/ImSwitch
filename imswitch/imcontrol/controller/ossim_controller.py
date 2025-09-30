"""Controller logic for orchestrating OSSIM acquisitions."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Tuple

import numpy as np
from qtpy import QtCore

from imswitch.imcontrol.client.ossim_client import OssimClient


@dataclass
class _FrameResult:
    """Container holding frame data and an optional frame index."""

    data: Optional[np.ndarray]
    index: Optional[int]


class _FunctionRunnable(QtCore.QRunnable):
    """Utility runnable executing a callable inside a QThreadPool."""

    def __init__(self, func: Callable[[], None]) -> None:
        super().__init__()
        self._func = func

    def run(self) -> None:  # pragma: no cover - executed in worker thread
        self._func()


class OssimController(QtCore.QObject):
    """Backend worker coordinating OSSIM pattern display and acquisition."""

    logMessage = QtCore.Signal(str)
    errorMessage = QtCore.Signal(str)
    imagesReady = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray)
    reconstructionReady = QtCore.Signal(np.ndarray)

    def __init__(
        self,
        detectors_manager,
        *,
        client: Optional[OssimClient] = None,
        base_url: str = "http://192.168.137.2:8000",
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        self._detectors_manager = detectors_manager
        self._client = client or OssimClient(base_url=base_url, timeout=timeout)
        self._thread_pool = QtCore.QThreadPool()
        self._stop_event = threading.Event()
        self._capture_running = threading.Event()
        self._latest_images: Tuple[
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
        ] = (None, None, None)

    @property
    def base_url(self) -> str:
        """Return the currently configured FastAPI base URL."""

        return self._client.base_url

    def display(self, pattern_id: int) -> None:
        """Display a single OSSIM pattern without blocking the UI."""

        if pattern_id not in {0, 1, 2}:
            self._emit_error(f"无效的图案 ID：{pattern_id}")
            return

        def task() -> None:
            self._emit_log(f"请求显示图案 {pattern_id}")
            response = self._client.display(pattern_id)
            if response.get("ok"):
                self._emit_log(f"图案 {pattern_id} 显示成功")
            else:
                self._emit_error(response.get("error", "未知错误"))

        self._thread_pool.start(_FunctionRunnable(task))

    def capture_three_phase(self, delay_s: float = 0.2) -> None:
        """Execute the three-pattern capture workflow on a worker thread."""

        if self._capture_running.is_set():
            self._emit_log("已有采集任务在进行，忽略新的启动请求")
            return

        self._capture_running.set()
        self._stop_event.clear()

        def task() -> None:
            try:
                self._emit_log("开始三相位采集")
                images = self._perform_three_phase_capture(delay_s)
                if images is not None:
                    self._latest_images = images
                    self.imagesReady.emit(*images)
                    self._emit_log("三相位采集完成")
            except Exception as exc:  # pragma: no cover - defensive guard
                self._emit_error(f"采集过程中出现异常：{exc}")
            finally:
                self._capture_running.clear()
                self._stop_event.clear()

        self._thread_pool.start(_FunctionRunnable(task))

    def stop(self) -> None:
        """Request the currently running workflow to stop."""

        if self._capture_running.is_set():
            self._emit_log("收到停止指令，正在终止采集")
            self._stop_event.set()
        else:
            self._emit_log("当前没有正在运行的采集任务")

    def reconstruct(
        self,
        image0: Optional[np.ndarray],
        image1: Optional[np.ndarray],
        image2: Optional[np.ndarray],
    ) -> None:
        """Perform classic IOS reconstruction in the background."""

        def task() -> None:
            try:
                if image0 is None or image1 is None or image2 is None:
                    raise ValueError("请先完成三张图像的采集")
                self._emit_log("开始进行三相位重建")
                reconstruction = self._compute_reconstruction(image0, image1, image2)
                self.reconstructionReady.emit(reconstruction)
                self._emit_log("重建完成")
            except Exception as exc:  # pragma: no cover - defensive guard
                self._emit_error(f"重建失败：{exc}")

        self._thread_pool.start(_FunctionRunnable(task))

    def _perform_three_phase_capture(
        self,
        delay_s: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        try:
            acq_handle = self._detectors_manager.startAcquisition()
        except Exception as exc:
            self._emit_error(f"启动相机采集失败：{exc}")
            return None

        images: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = (
            None,
            None,
            None,
        )

        try:
            for idx in (0, 1, 2):
                if self._stop_event.is_set():
                    self._emit_log("采集已被用户中断")
                    return None

                if not self._display_and_confirm(idx):
                    return None

                if self._stop_event.is_set():
                    self._emit_log("采集已被用户中断")
                    return None

                frame = self._capture_single_frame()
                if frame is None:
                    return None

                images = tuple(frame if i == idx else images[i] for i in range(3))  # type: ignore[misc]

                if idx < 2:
                    self._sleep_with_stop(delay_s)

            return images if all(img is not None for img in images) else None
        finally:
            try:
                self._detectors_manager.stopAcquisition(acq_handle)
            except Exception:
                pass

    def _display_and_confirm(self, pattern_id: int) -> bool:
        response = self._client.display(pattern_id)
        if not response.get("ok"):
            self._emit_error(response.get("error", f"显示图案 {pattern_id} 失败"))
            return False
        self._emit_log(f"图案 {pattern_id} 已显示，等待相机帧")
        return True

    def _capture_single_frame(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        detector = self._detectors_manager.getCurrentDetector()

        frame = self._try_direct_snap(detector)
        if frame is not None:
            return frame

        initial = self._get_frame(detector)
        deadline = time.monotonic() + timeout

        while not self._stop_event.is_set() and time.monotonic() < deadline:
            wait_time = deadline - time.monotonic()
            if wait_time <= 0:
                break
            if not self._wait_for_new_frame(wait_time):
                break
            current = self._get_frame(detector)
            if current.data is None:
                continue
            if self._is_new_frame(initial, current):
                return current.data

        if self._stop_event.is_set():
            self._emit_log("采集流程因停止指令提前结束")
            return None

        self._emit_error("等待相机返回帧超时，请检查曝光或连接状态")
        return None

    def _wait_for_new_frame(self, timeout: float) -> bool:
        if timeout <= 0:
            return False

        event = threading.Event()

        def on_new_frame() -> None:
            event.set()

        try:
            self._detectors_manager.sigNewFrame.connect(
                on_new_frame,
                QtCore.Qt.ConnectionType.QueuedConnection,
            )
        except TypeError:
            self._detectors_manager.sigNewFrame.connect(on_new_frame)

        triggered = event.wait(timeout)

        try:
            self._detectors_manager.sigNewFrame.disconnect(on_new_frame)
        except Exception:
            pass

        return triggered

    def _get_frame(self, detector) -> _FrameResult:
        try:
            frame, index = detector.getLatestFrame(returnFrameNumber=True)
            return _FrameResult(self._ensure_gray(frame), int(index) if index is not None else None)
        except TypeError:
            try:
                frame = detector.getLatestFrame()
            except Exception:
                frame = None
        except AttributeError:
            frame = detector.getLatestFrame()
        except Exception:
            frame = None
        return _FrameResult(self._ensure_gray(frame), None)

    @staticmethod
    def _is_new_frame(initial: _FrameResult, current: _FrameResult) -> bool:
        if current.data is None:
            return False
        if current.index is not None and initial.index is not None:
            return current.index != initial.index
        if initial.data is None:
            return True
        try:
            return not np.array_equal(initial.data, current.data)
        except Exception:
            return True

    def _try_direct_snap(self, detector) -> Optional[np.ndarray]:
        methods = [
            "snap",
            "snap_image",
            "snapImage",
            "snapImagelepmonCam",
            "get_snap",
        ]
        for name in methods:
            func = getattr(detector, name, None)
            if func is None or not callable(func):
                continue
            try:
                result = func()
            except TypeError:
                continue
            except Exception as exc:
                self._emit_log(f"直接抓拍方法 {name} 失败：{exc}")
                continue
            frame = self._ensure_gray(result)
            if frame is not None:
                return frame
        return None

    def _compute_reconstruction(
        self,
        i0: np.ndarray,
        i1: np.ndarray,
        i2: np.ndarray,
    ) -> np.ndarray:
        i0f = self._ensure_gray(i0)
        i1f = self._ensure_gray(i1)
        i2f = self._ensure_gray(i2)
        if i0f is None or i1f is None or i2f is None:
            raise ValueError("提供的图像数据无效")

        r = np.sqrt(
            np.square(i0f - i1f)
            + np.square(i0f - i2f)
            + np.square(i1f - i2f)
        ).astype(np.float32)

        min_val = float(np.nanmin(r))
        max_val = float(np.nanmax(r))
        if math.isfinite(min_val) and math.isfinite(max_val) and max_val > min_val:
            r = (r - min_val) / (max_val - min_val)

        return r

    def _sleep_with_stop(self, duration: float) -> None:
        deadline = time.monotonic() + max(0.0, duration)
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(min(0.05, deadline - time.monotonic()))

    def _ensure_gray(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim >= 3:
            arr = arr.mean(axis=-1)
        return arr.astype(np.float32, copy=False)

    def _emit_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logMessage.emit(f"[{timestamp}] {message}")

    def _emit_error(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = f"[{timestamp}] {message}"
        self.logMessage.emit(text)
        self.errorMessage.emit(text)
