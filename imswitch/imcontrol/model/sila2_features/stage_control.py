"""
SiLA2 Stage Control Feature for OpenUC2 ImSwitch.

Provides SiLA2 commands and properties for microscope stage positioning.
"""

import abc

from ._compat import (
    SilaFeatureBase,
    UnobservableProperty,
    UnobservableCommand,
    Response,
)


class StageControlFeature(SilaFeatureBase, metaclass=abc.ABCMeta):
    """
    Stage Control Feature

    Provides stage positioning commands for the OpenUC2 microscope.
    Supports reading current position and moving to absolute coordinates.
    """

    def __init__(self):
        super().__init__(
            originator="org.openuc2",
            category="microscopy",
            version="1.0",
            maturity_level="Draft",
        )

    @abc.abstractmethod
    @UnobservableProperty()
    async def get_stage_position(self) -> str:
        """
        Retrieve the current XYZ position of the stage.

        .. return:: Comma-separated string of X,Y,Z coordinates in micrometers.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def move_stage_to(
        self,
        x_um: float,
        y_um: float,
        z_um: float = 0.0,
        speed: float = 10000.0,
        is_blocking: bool = True,
    ) -> bool:
        """
        Move the stage to an absolute XYZ position.

        .. parameter:: x_um: Target X position in micrometers.
        .. parameter:: y_um: Target Y position in micrometers.
        .. parameter:: z_um: Target Z position in micrometers (0 = keep current).
        .. parameter:: speed: Movement speed in units per second.
        .. parameter:: is_blocking: Whether to wait for the movement to complete.
        .. return:: True if successful, False otherwise.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def move_stage_relative(
        self,
        dx_um: float = 0.0,
        dy_um: float = 0.0,
        dz_um: float = 0.0,
        speed: float = 10000.0,
        is_blocking: bool = True,
    ) -> bool:
        """
        Move the stage by a relative offset.

        .. parameter:: dx_um: Relative X displacement in micrometers.
        .. parameter:: dy_um: Relative Y displacement in micrometers.
        .. parameter:: dz_um: Relative Z displacement in micrometers.
        .. parameter:: speed: Movement speed in units per second.
        .. parameter:: is_blocking: Whether to wait for the movement to complete.
        .. return:: True if successful, False otherwise.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def home_stage(self, is_blocking: bool = True) -> bool:
        """
        Home the stage (move to loading/home position).

        .. parameter:: is_blocking: Whether to wait for the homing to complete.
        .. return:: True if successful, False otherwise.
        """
