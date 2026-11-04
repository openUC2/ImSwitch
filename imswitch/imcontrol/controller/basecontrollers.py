from imswitch.imcommon.controller import WidgetController, WidgetControllerFactory


class ImConWidgetControllerFactory(WidgetControllerFactory):
    """ Factory class for creating a ImConWidgetController object. """

    def __init__(self, setupInfo, master, commChannel, moduleCommChannel):
        super().__init__(setupInfo=setupInfo, master=master, commChannel=commChannel,
                         moduleCommChannel=moduleCommChannel)


class ImConWidgetController(WidgetController):
    """ Superclass for all ImConWidgetController.
    All WidgetControllers should have access to the setup information,
    MasterController, CommunicationChannel and the linked Widget. """

    def __init__(self, setupInfo, commChannel, master, *args, **kwargs):
        # Protected attributes, which should only be accessed from controller and its subclasses
        self._setupInfo = setupInfo
        self._commChannel = commChannel
        self._master = master

        # Init superclass
        super().__init__(*args, **kwargs)


class LiveUpdatedController(ImConWidgetController):
    """ Superclass for those controllers that will update the widgets with an
    upcoming frame from the camera.  Should be either active or not, and have
    an update function. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = False

    def update(self, detectorName, im, init, isCurrentDetector):
        raise NotImplementedError


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
