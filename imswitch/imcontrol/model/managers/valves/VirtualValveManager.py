"""A valve without hardware; it remembers each switch for jobs and tests."""
from .ValveManager import ValveManager


class VirtualValveManager(ValveManager):
    def __init__(self, valveInfo, name, **_lowLevelManagers):
        super().__init__(valveInfo, name)
        self.history = []

    def _apply(self, setting):
        self.history.append(setting)
