import numpy as np
from qtpy import QtCore, QtWidgets, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from imswitch import IS_HEADLESS

from PyQt5 import QtGui, QtWidgets
import PyQt5
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.view import guitools
from .basewidgets import NapariHybridWidget
import os



class WorkflowWidget(NapariHybridWidget):
    """ Widget containing Workflow interface. """

    def __post_init__(self):
        #super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
