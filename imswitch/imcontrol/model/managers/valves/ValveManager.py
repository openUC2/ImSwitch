"""Valves: named positions (``managerProperties.positions``) and what the
manager sends for each."""
from abc import ABC, abstractmethod


class ValveManager(ABC):
    """Base for one valve of the ``valves`` setup block."""

    def __init__(self, valveInfo, name: str):
        self.name = name
        self.properties = dict(valveInfo.managerProperties or {})
        self._positions = dict(self.properties.get("positions") or {})
        if not self._positions:
            raise ValueError(f"valve {name}: managerProperties.positions is empty")
        self.position = None

    @property
    def positions(self):
        return list(self._positions)

    def set_position(self, position: str) -> None:
        if position not in self._positions:
            raise ValueError(f"valve {self.name} has no position {position!r}; "
                             f"it has {self.positions}")
        self._apply(self._positions[position])
        self.position = position

    def finalize(self) -> None:
        pass

    @abstractmethod
    def _apply(self, setting) -> None: ...
