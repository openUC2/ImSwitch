from qtpy import QtCore, QtWidgets
import numpy as np
from .basewidgets import NapariHybridWidget


class DMDWidget(NapariHybridWidget):
    """Simple DMD control panel to talk to a Raspberry Pi FastAPI service.

    Controls:
    - Host field (default http://192.168.137.2:8000)
    - Buttons to display pattern 0/1/2
    - Button to run synchronized 3-shot capture (DMD 0->1->2 with camera)
    - Button to run reconstruction (Classic IOS) from the last 3 images
    """

    sigDisplayPattern = QtCore.Signal(int)
    sigRunThreeShot = QtCore.Signal()
    sigReconstruct = QtCore.Signal()
    sigCheckStatus = QtCore.Signal()

    def __post_init__(self):
        self.setObjectName("DMDWidget")
        self._lastLayer = None

        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)

        # Host input
        self.hostEdit = QtWidgets.QLineEdit("http://192.168.137.2:8000")
        self.hostEdit.setToolTip("Raspberry Pi FastAPI base URL")
        grid.addWidget(QtWidgets.QLabel("DMD FastAPI URL"), 0, 0, 1, 1)
        grid.addWidget(self.hostEdit, 0, 1, 1, 3)

        # Status check
        self.btnCheck = QtWidgets.QPushButton("Check Status")
        self.btnCheck.setToolTip("Call /health to verify DMD connectivity")
        grid.addWidget(self.btnCheck, 1, 0)

        # Pattern display buttons
        self.btnShow0 = QtWidgets.QPushButton("Show 0")
        self.btnShow1 = QtWidgets.QPushButton("Show 1")
        self.btnShow2 = QtWidgets.QPushButton("Show 2")
        grid.addWidget(self.btnShow0, 1, 1)
        grid.addWidget(self.btnShow1, 1, 2)
        grid.addWidget(self.btnShow2, 1, 3)

        # 3-shot + reconstruction
        self.btnThreeShot = QtWidgets.QPushButton("Run 3-shot (0-1-2)")
        self.btnReconstruct = QtWidgets.QPushButton("Reconstruction (IOS)")
        grid.addWidget(self.btnThreeShot, 2, 0, 1, 2)
        grid.addWidget(self.btnReconstruct, 2, 2)

        # Status
        self.statusLabel = QtWidgets.QLabel("")
        grid.addWidget(self.statusLabel, 3, 0, 1, 4)

        # Connections
        self.btnCheck.clicked.connect(self.sigCheckStatus.emit)
        self.btnShow0.clicked.connect(lambda: self.sigDisplayPattern.emit(0))
        self.btnShow1.clicked.connect(lambda: self.sigDisplayPattern.emit(1))
        self.btnShow2.clicked.connect(lambda: self.sigDisplayPattern.emit(2))
        self.btnThreeShot.clicked.connect(self.sigRunThreeShot.emit)
        self.btnReconstruct.clicked.connect(self.sigReconstruct.emit)

    # UI helpers
    def getHost(self) -> str:
        return self.hostEdit.text().strip()

    def setStatus(self, text: str):
        self.statusLabel.setText(text)

    def showImage(self, im: np.ndarray, name: str = "DMD Reconstruction"):
        """Display or update an image in napari."""
        if self._lastLayer is None or name not in self.viewer.layers:
            # Create new layer
            self._lastLayer = self.viewer.add_image(
                im, name=name, rgb=False, blending="translucent", colormap="gray"
            )
        else:
            self._lastLayer.data = im

        # Autoscale a bit
        try:
            vmin = float(np.min(im))
            vmax = float(np.max(im))
            if vmax > vmin:
                self._lastLayer.contrast_limits = (vmin, vmax)
        except Exception:
            pass


# Copyright (C) 2020-2025 ImSwitch developers
# GPLv3 License
