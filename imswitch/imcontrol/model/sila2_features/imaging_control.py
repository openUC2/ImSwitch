"""
SiLA2 Imaging Control Feature for OpenUC2 ImSwitch.

Provides SiLA2 commands and properties for detector and illumination control.
"""

import abc

from ._compat import (
    SilaFeatureBase,
    UnobservableProperty,
    UnobservableCommand,
    Response,
)


class ImagingControlFeature(SilaFeatureBase, metaclass=abc.ABCMeta):
    """
    Imaging Control Feature

    Provides detector frame acquisition and illumination control
    for the OpenUC2 microscope.
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
    async def get_available_detectors(self) -> str:
        """
        Get a comma-separated list of available detector names.

        .. return:: Comma-separated detector names.
        """

    @abc.abstractmethod
    @UnobservableProperty()
    async def get_available_illumination_sources(self) -> str:
        """
        Get a comma-separated list of available illumination source names.

        .. return:: Comma-separated illumination source names.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="FrameBase64")
    async def snap_image(
        self,
        detector_name: str = "",
        exposure_time_ms: float = -1.0,
        gain: float = -1.0,
    ) -> str:
        """
        Capture a single frame from the specified detector.

        Returns the image as a base64-encoded PNG string.

        .. parameter:: detector_name: Name of the detector (empty = first available).
        .. parameter:: exposure_time_ms: Exposure time in ms (-1 = use current setting).
        .. parameter:: gain: Camera gain (-1 = use current setting).
        .. return:: Base64-encoded PNG image data.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def set_illumination(
        self,
        channel_name: str,
        intensity: float,
        enabled: bool = True,
    ) -> bool:
        """
        Set illumination channel intensity and state.

        .. parameter:: channel_name: Name of the illumination channel.
        .. parameter:: intensity: Intensity value.
        .. parameter:: enabled: Whether the channel should be enabled.
        .. return:: True if successful, False otherwise.
        """

    @abc.abstractmethod
    @UnobservableCommand()
    @Response(name="Result")
    async def set_exposure_time(
        self,
        exposure_time_ms: float,
        detector_name: str = "",
    ) -> bool:
        """
        Set the exposure time for a detector.

        .. parameter:: exposure_time_ms: Exposure time in milliseconds.
        .. parameter:: detector_name: Name of the detector (empty = first available).
        .. return:: True if successful, False otherwise.
        """
