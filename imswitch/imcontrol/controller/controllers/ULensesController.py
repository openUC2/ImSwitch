import numpy as np

from ..basecontrollers import ImConWidgetController


class ULensesController(ImConWidgetController):
    """ Linked to ULensesWidget. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Connect ULensesWidget signals
        self._widget.sigULensesClicked.connect(self.updateGrid)
        self._widget.sigUShowLensesChanged.connect(self.toggleULenses)

    def updateGrid(self):
        """ Updates plot with new parameters. """
        x, y, px, upx, upy = self._widget.getParameters()
        size_x, size_y = self._master.detectorsManager.execOnCurrent(lambda c: c.shape)
        pattern_x = np.arange(x, size_x, upx / px)
        pattern_y = np.arange(y, size_y, upy / px)
        grid = np.array(np.meshgrid(pattern_x, pattern_y)).T.reshape(-1, 2)
        self._widget.setData(x=grid[:, 0], y=grid[:, 1])

    def toggleULenses(self, show):
        """ Shows or hides grid. """
        self._widget.setULensesVisible(show)


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
