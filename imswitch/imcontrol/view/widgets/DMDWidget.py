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
        # Timing (ms) between pattern changes
        grid.addWidget(QtWidgets.QLabel("Delay (ms)"), 2, 0)
        self.delaySpin = QtWidgets.QSpinBox()
        self.delaySpin.setRange(0, 10000)
        self.delaySpin.setSingleStep(10)
        self.delaySpin.setValue(200)  # default 0.2 s
        self.delaySpin.setToolTip("Pattern切换之间的延迟 (毫秒). 原默认 200 ms")
        grid.addWidget(self.delaySpin, 2, 1)

        # Gaussian sigma for reconstruction
        grid.addWidget(QtWidgets.QLabel("Sigma"), 2, 2)
        self.sigmaSpin = QtWidgets.QDoubleSpinBox()
        self.sigmaSpin.setDecimals(2)
        self.sigmaSpin.setRange(0.0, 50.0)
        self.sigmaSpin.setSingleStep(0.25)
        self.sigmaSpin.setValue(3.0)
        self.sigmaSpin.setToolTip("重建后应用的高斯滤波 σ (像素). 设为0关闭滤波")
        grid.addWidget(self.sigmaSpin, 2, 3)

        # Row 3 already used for buttons; add toggles on next row
        # Toggles: Gaussian enable, Widefield enable, Export enable + filename
        self.chkGaussian = QtWidgets.QCheckBox("Use Gaussian")
        self.chkGaussian.setChecked(True)
        self.chkGaussian.setToolTip("是否应用高斯滤波 (sigma>0 时有效)")
        grid.addWidget(self.chkGaussian, 3, 3)

        self.chkWidefield = QtWidgets.QCheckBox("Show Widefield")
        self.chkWidefield.setChecked(True)
        self.chkWidefield.setToolTip("是否显示三帧平均widefield图")
        grid.addWidget(self.chkWidefield, 4, 0)

        self.chkExport = QtWidgets.QCheckBox("Export")
        self.chkExport.setChecked(False)
        self.chkExport.setToolTip("重建后导出 TIFF (IOS 以及可选 Widefield)")
        grid.addWidget(self.chkExport, 4, 1)

        self.chkExportRaw = QtWidgets.QCheckBox("Raw 3 frames")
        self.chkExportRaw.setChecked(False)
        self.chkExportRaw.setToolTip("同时导出三张原始帧 I1/I2/I3 (原始数据类型)")
        grid.addWidget(self.chkExportRaw, 4, 2)

        self.exportNameEdit = QtWidgets.QLineEdit("dmd_recon")
        self.exportNameEdit.setToolTip("导出文件基础名（不含扩展名），存放于 recording 目录 (若存在) 或当前目录")
        grid.addWidget(QtWidgets.QLabel("Base Name"), 5, 0)
        grid.addWidget(self.exportNameEdit, 5, 1, 1, 3)

        # Export directory selection
        default_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation)
        candidate = QtCore.QDir.toNativeSeparators(default_dir + "/ImSwitchConfig/recordings")
        self.exportDirEdit = QtWidgets.QLineEdit(candidate)
        self.exportDirEdit.setToolTip("导出目录，可手动修改。默认 文档/ImSwitchConfig/recordings")
        self.btnBrowseDir = QtWidgets.QPushButton("...")
        self.btnBrowseDir.setFixedWidth(30)
        grid.addWidget(QtWidgets.QLabel("Dir"), 6, 0)
        grid.addWidget(self.exportDirEdit, 6, 1, 1, 2)
        grid.addWidget(self.btnBrowseDir, 6, 3)

        # 3-shot + reconstruction
        self.btnThreeShot = QtWidgets.QPushButton("Run 3-shot (0-1-2)")
        self.btnReconstruct = QtWidgets.QPushButton("Reconstruction (IOS)")
        grid.addWidget(self.btnThreeShot, 3, 0, 1, 2)
        grid.addWidget(self.btnReconstruct, 3, 2)

        # Status
        self.statusLabel = QtWidgets.QLabel("")
        grid.addWidget(self.statusLabel, 6, 0, 1, 4)

        # Connections
        self.btnCheck.clicked.connect(self.sigCheckStatus.emit)
        self.btnShow0.clicked.connect(lambda: self.sigDisplayPattern.emit(0))
        self.btnShow1.clicked.connect(lambda: self.sigDisplayPattern.emit(1))
        self.btnShow2.clicked.connect(lambda: self.sigDisplayPattern.emit(2))
        self.btnThreeShot.clicked.connect(self.sigRunThreeShot.emit)
        self.btnReconstruct.clicked.connect(self.sigReconstruct.emit)
        self.btnBrowseDir.clicked.connect(self._browseExportDir)

    # UI helpers
    def getHost(self) -> str:
        return self.hostEdit.text().strip()

    def getDelaySeconds(self) -> float:
        """Return inter-pattern delay in seconds (spin box is ms)."""
        return self.delaySpin.value() / 1000.0

    def getSigma(self) -> float:
        return float(self.sigmaSpin.value())

    def isGaussianEnabled(self) -> bool:
        return self.chkGaussian.isChecked()

    def isWidefieldEnabled(self) -> bool:
        return self.chkWidefield.isChecked()

    def isExportEnabled(self) -> bool:
        return self.chkExport.isChecked()

    def isExportRawEnabled(self) -> bool:
        return self.chkExportRaw.isChecked()

    def getExportBaseName(self) -> str:
        return self.exportNameEdit.text().strip() or "dmd_recon"

    def getExportDirectory(self) -> str:
        return self.exportDirEdit.text().strip()

    def _browseExportDir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出目录", self.getExportDirectory() or "")
        if d:
            self.exportDirEdit.setText(d)

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