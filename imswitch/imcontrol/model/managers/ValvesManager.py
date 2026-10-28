from .MultiManager import MultiManager


class ValvesManager(MultiManager):
    """The ``valves`` setup block: one ValveManager per named valve."""

    def __init__(self, valveInfos, **lowLevelManagers):
        super().__init__(valveInfos, 'valves', **lowLevelManagers)
