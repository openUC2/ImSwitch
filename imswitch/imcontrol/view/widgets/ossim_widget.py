"""Dockable widget for OSSIM three-phase acquisition."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from datetime import datetime
from qtpy import QtCore, QtWidgets

from imswitch.imcontrol.controller.ossim_controller import OssimController
from .basewidgets import NapariHybridWidget


class OssimWidget(NapariHybridWidget):
    """Dock widget that controls OSSIM pattern display and acquisition."""

    DEFAULT_URL = "http://192.168.137.2:8000"

    def __post_init__(self) -> None:
        self._controller: Optional[OssimController] = None
        self._controller_log_proxy = None
        self._images: Dict[int, Optional[np.ndarray]] = {0: None, 1: None, 2: None}
        self._capture_running = False
        self._build_ui()
        self._set_buttons_enabled(False)
        self._append_log("等待控制器初始化…")

    def bind_controller(self, controller: OssimController) -> None:
        if self._controller is not None and self._controller_log_proxy is not None:
            try:
                self._controller.logMessage.disconnect(self._controller_log_proxy)
            except Exception:
                pass
        self._controller = controller
        self._controller_log_proxy = lambda msg: self._append_log(msg, raw=True)
        controller.logMessage.connect(self._controller_log_proxy)
        controller.errorMessage.connect(self._handle_error)
        controller.imagesReady.connect(self._handle_images)
        controller.reconstructionReady.connect(self._handle_reconstruction)
        self.url_edit.setText(controller.base_url)
        self._set_buttons_enabled(True)
        self._append_log("OSSIM 控制器已就绪")

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        url_layout = QtWidgets.QFormLayout()
        self.url_edit = QtWidgets.QLineEdit(self.DEFAULT_URL)
        self.url_edit.setReadOnly(True)
        url_layout.addRow("树莓派地址", self.url_edit)
        layout.addLayout(url_layout)

        button_row = QtWidgets.QHBoxLayout()
        self.display_buttons = []
        for idx in range(3):
            btn = QtWidgets.QPushButton(f"Display {idx}")
            btn.clicked.connect(lambda _=False, i=idx: self._display_pattern(i))
            button_row.addWidget(btn)
            self.display_buttons.append(btn)
        layout.addLayout(button_row)

        control_row = QtWidgets.QHBoxLayout()
        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 2.0)
        self.delay_spin.setSingleStep(0.05)
        self.delay_spin.setValue(0.2)
        self.delay_spin.setSuffix(" s")
        control_row.addWidget(QtWidgets.QLabel("延迟"))
        control_row.addWidget(self.delay_spin)

        self.start_button = QtWidgets.QPushButton("Start 3-phase Capture")
        self.start_button.clicked.connect(self._start_capture)
        control_row.addWidget(self.start_button)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_capture)
        control_row.addWidget(self.stop_button)

        self.reconstruct_button = QtWidgets.QPushButton("Reconstruction")
        self.reconstruct_button.clicked.connect(self._run_reconstruction)
        control_row.addWidget(self.reconstruct_button)

        layout.addLayout(control_row)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(160)
        layout.addWidget(self.log_edit)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in self.display_buttons:
            btn.setEnabled(enabled and not self._capture_running)
        self.start_button.setEnabled(enabled and not self._capture_running)
        self.reconstruct_button.setEnabled(enabled and self._has_images() and not self._capture_running)
        self.stop_button.setEnabled(enabled and self._capture_running)

    def _display_pattern(self, pattern_id: int) -> None:
        if self._controller is None:
            self._append_log("控制器尚未连接", error=True)
            return
        self._controller.display(pattern_id)

    def _start_capture(self) -> None:
        if self._controller is None:
            self._append_log("控制器尚未连接", error=True)
            return
        self._capture_running = True
        self._set_buttons_enabled(True)
        self._controller.capture_three_phase(self.delay_spin.value())

    def _stop_capture(self) -> None:
        if self._controller is None:
            return
        self._controller.stop()
        self._capture_running = False
        self._set_buttons_enabled(True)

    def _run_reconstruction(self) -> None:
        if self._controller is None:
            return
        if not self._has_images():
            self._append_log("请先完成一次三相位采集", error=True)
            return
        self._controller.reconstruct(self._images[0], self._images[1], self._images[2])

    def _handle_images(self, i0: np.ndarray, i1: np.ndarray, i2: np.ndarray) -> None:
        self._images = {0: i0, 1: i1, 2: i2}
        self._update_layer("OSSIM I0", i0)
        self._update_layer("OSSIM I1", i1)
        self._update_layer("OSSIM I2", i2)
        self._capture_running = False
        self._set_buttons_enabled(True)
        self._append_log("三相位图像已更新")

    def _handle_reconstruction(self, image: np.ndarray) -> None:
        self._update_layer("OSSIM Reconstruction", image)
        self._append_log("重建图像已更新")

    def _handle_error(self, message: str) -> None:
        self._append_log(message, raw=True)
        if self._capture_running:
            self._capture_running = False
            self._set_buttons_enabled(True)

    def _update_layer(self, name: str, data: np.ndarray) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_image(data, name=name, blending="additive")

    def _append_log(self, message: str, *, error: bool = False, raw: bool = False) -> None:
        if raw:
            text = message
        else:
            prefix = datetime.now().strftime('[%H:%M:%S] ')
            text = f"{prefix}{message}"
            if error:
                text = f"{text} (错误)"
        self.log_edit.appendPlainText(text)

    def _has_images(self) -> bool:
        return all(img is not None for img in self._images.values())

    def closeEvent(self, event) -> None:
        self._capture_running = False
        self._set_buttons_enabled(self._controller is not None)
        super().closeEvent(event)

