# -*- coding: utf-8 -*-
from qtpy import QtWidgets

class OSSIMWidget(QtWidgets.QWidget):
    displayName = "OSSIM"  # 必须与 setup 里的 availableWidgets 一致

    def __init__(self, *args, **kwargs):
        # 工厂会传各种 kw，比如 napariViewer/parent/options/setupInfo/...
        parent = kwargs.pop("parent", None)
        # 把可能用到的对象先收起来（名称跟随 ImSwitch 习惯）
        self.napariViewer = kwargs.pop("napariViewer", None)
        self.mainView     = kwargs.pop("mainView", None)
        self.options      = kwargs.pop("options", None)
        self.setupInfo    = kwargs.pop("setupInfo", None)
        # 其余未知 kw 丢弃，避免传进 Qt
        kwargs.clear()

        super().__init__(parent)
        self.controller = None  # setController() 注入

        self._build()  # 先搭 UI（此时不绑定事件）

    def setController(self, controller):
        """工厂在创建后会调用这个注入 controller。"""
        self.controller = controller
        self._wire()
        self._refresh()

    # ---------- UI ----------
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        self.lbl = QtWidgets.QLabel("Server: ?")
        lay.addWidget(self.lbl)

        row = QtWidgets.QHBoxLayout(); lay.addLayout(row)
        row.addWidget(QtWidgets.QLabel("Pause (s):"))
        self.spinPause = QtWidgets.QDoubleSpinBox()
        self.spinPause.setRange(0, 10)
        self.spinPause.setDecimals(3)
        self.spinPause.setValue(0.2)
        row.addWidget(self.spinPause)
        self.btnSetPause = QtWidgets.QPushButton("Set pause"); row.addWidget(self.btnSetPause)

        row2 = QtWidgets.QHBoxLayout(); lay.addLayout(row2)
        self.btnStart = QtWidgets.QPushButton("Start ∞")
        self.btnStartN = QtWidgets.QPushButton("Start N")
        self.spinN = QtWidgets.QSpinBox(); self.spinN.setRange(1, 10000); self.spinN.setValue(5)
        self.btnStop = QtWidgets.QPushButton("Stop")
        row2.addWidget(self.btnStart); row2.addWidget(self.btnStartN); row2.addWidget(self.spinN); row2.addWidget(self.btnStop)

        row3 = QtWidgets.QHBoxLayout(); lay.addLayout(row3)
        row3.addWidget(QtWidgets.QLabel("Display idx:"))
        self.spinIdx = QtWidgets.QSpinBox(); self.spinIdx.setRange(0, 9999)
        self.btnDisplay = QtWidgets.QPushButton("Display")
        row3.addWidget(self.spinIdx); row3.addWidget(self.btnDisplay)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(100)
        lay.addWidget(self.log)
        lay.addStretch(1)

    def _wire(self):
        if self.controller is None:
            return
        self.btnSetPause.clicked.connect(lambda: self._call(self.controller.set_pause, self.spinPause.value()))
        self.btnStart.clicked.connect(lambda: self._call(self.controller.start, -1))
        self.btnStartN.clicked.connect(lambda: self._call(self.controller.start, self.spinN.value()))
        self.btnStop.clicked.connect(lambda: self._call(self.controller.stop))
        self.btnDisplay.clicked.connect(lambda: self._call(self.controller.display, self.spinIdx.value()))
        if hasattr(self.controller, "sigStatus"):
            self.controller.sigStatus.connect(lambda st: self._append(f"status: {st}"))

    def _refresh(self):
        if self.controller is None:
            self.lbl.setText("Server: (no controller)")
            return
        try:
            self.lbl.setText(f"Server: {self.controller.health()}")
        except Exception as e:
            self.lbl.setText(f"Server: ERR {e}")

    def _append(self, s: str):
        self.log.appendPlainText(str(s))

    def _call(self, fn, *a):
        try:
            self._append(fn(*a))
        except Exception as e:
            self._append(f"ERR: {e}")
