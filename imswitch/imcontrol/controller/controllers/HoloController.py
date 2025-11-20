"""
DEPRECATED: This module is deprecated and split into two separate controllers:
- InLineHoloController: For inline holography processing
- OffAxisHoloController: For off-axis holography processing

Please use those controllers instead.
For backward compatibility, this module imports InLineHoloController as HoloController.
"""

from .InLineHoloController import InLineHoloController

# Backward compatibility: alias InLineHoloController as HoloController
HoloController = InLineHoloController

__all__ = ['HoloController', 'InLineHoloController']


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
