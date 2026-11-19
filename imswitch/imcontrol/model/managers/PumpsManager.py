from .MultiManager import MultiManager


class PumpsManager(MultiManager):
    """The ``pumps`` setup block: one PumpManager per named pump."""

    def __init__(self, pumpInfos, **lowLevelManagers):
        super().__init__(pumpInfos, 'pumps', **lowLevelManagers)
