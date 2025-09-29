# -*- coding: utf-8 -*-
from imswitch.imcontrol.controller.basecontrollers import ImConWidgetController
from imswitch.imcontrol.model.managers.OSSIMManager import OSSIMManager
from imswitch.imcommon.framework import Signal

class OSSIMController(ImConWidgetController):
    """
    Controller: glue between Manager and Widget.
    NOTE: no 'from imswitch.imcontrol.model import model' needed.
    """
    def __init__(self, *args, host: str = "192.168.137.2", port: int = 8000,
                 default_display_time: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = OSSIMManager(host=host, port=port, default_display_time=default_display_time)
        self.sigStatus = self.manager.sigStatus  # forward status to widget

    # methods the widget will call
    def health(self): return self.manager.health()
    def list(self): return self.manager.list_patterns()
    def set_pause(self, s: float): return self.manager.set_pause(s)
    def display(self, i: int): return self.manager.display(i)
    def start(self, cycles: int = -1): return self.manager.start(cycles)
    def stop(self): return self.manager.stop()
