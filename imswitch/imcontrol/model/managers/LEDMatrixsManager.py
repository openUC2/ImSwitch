from .MultiManager import MultiManager


class LEDMatrixsManager(MultiManager):
    """ LEDMatrixsManager is an interface for dealing with LEDMatrixManagers. It is a
    MultiManager for LEDMatrixs. """

    def __init__(self, LEDMatrixInfos, **lowLevelManagers):
        super().__init__(LEDMatrixInfos, 'LEDMatrixs', **lowLevelManagers)


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
