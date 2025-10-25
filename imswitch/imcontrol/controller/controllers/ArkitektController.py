from imswitch.imcontrol.model.managers.ArkitektManager import set_global_context_locally
from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import dirtools, initLogger, APIExport
import xarray as xr
from arkitekt_next import register, easy, progress
from mikro_next.api.schema import (
    Image,
    from_array_like,
    create_stage,
    PartialAffineTransformationViewInput,
)
from typing import Generator


# =========================
# Controller
# =========================
class ArkitektController(ImConWidgetController):
    """
    Controller for the Arkitekt widget.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._logger.debug("Initializing")

        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        if len(allDetectorNames) == 0:
            return
        self.mDetector = self._master.detectorsManager[
            self._master.detectorsManager.getAllDeviceNames()[0]
        ]

        self.arkitekt_app = self._master.arkitektManager.get_arkitekt_app()
        self.arkitekt_app.register(self.moveToSampleLoadingPosition)
        self.arkitekt_app.register(self.runTileScan)
        self.arkitekt_app.run_detached()

    def moveToSampleLoadingPosition(
        self, speed: float = 10000, is_blocking: bool = True
    ):
        """Move to sample loading position."""
        positionerNames = self._master.positionersManager.getAllDeviceNames()
        if len(positionerNames) == 0:
            self._logger.warning(
                "No positioners available to move to sample loading position."
            )
            return
        positionerName = positionerNames[0]
        self._logger.debug(
            f"Moving to sample loading position for positioner {positionerName}"
        )
        self._master.positionersManager[positionerName].moveToSampleLoadingPosition(
            speed=speed, is_blocking=is_blocking
        )

    def runTileScan(
        self,
        xRange: float = 100,
        yRange: float = 100,
        xStep: int = 10,
        yStep: int = 10,
        speed: float = 10000,
        positionerName: str | None = None,
        performAutofocus: bool = False,
        autofocusRange: float = 100,
        autofocusResolution: float = 10,
    ) -> Generator[Image, None, None]:
        """Run a tile scan.

        Runs a tile scan by moving the specified positioner in a grid pattern,
        capturing images at each position, and yielding the images with appropriate
        affine transformations for stitching.

        Args:
            xRange (float): Total range to scan in the X direction.
            yRange (float): Total range to scan in the Y direction.
            xStep (int): Step size in the X direction.
            yStep (int): Step size in the Y direction.
            speed (float): Speed of the positioner movement.
            positionerName (str | None): Name of the positioner to use. If None
                the first available positioner will be used.
            performAutofocus (bool): Whether to perform autofocus at each tile position.
            autofocusRange (float): Range for autofocus scan in Z direction.
            autofocusResolution (float): Step size for autofocus scan.
        Yields:
            Image: Captured image with affine transformation for stitching.
        """
        # TODO: Implement a check if the positioner supports tile scanning
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._logger.debug(f"Starting tile scan for positioner {positionerName}")
        
        # Get autofocus controller if needed
        autofocusController = None
        if performAutofocus:
            autofocusController = self._master.getController('Autofocus')
            if autofocusController is None:
                self._logger.warning("Autofocus requested but AutofocusController not available")
                performAutofocus = False
        
        # have a tile scan function in the positioner manager inside a for loop
        mFrameList = []
        mPositioner = self._master.positionersManager[positionerName]

        stage = create_stage(name="Tile Scan Stage")

        for y in range(0, yRange, yStep):
            for x in range(0, xRange, xStep):
                mPositioner.move(value=(x, y), axis="XY", is_absolute=True)
                
                # Perform autofocus at this position if requested
                if performAutofocus and autofocusController is not None:
                    try:
                        # Call autofocus directly - no signals needed!
                        autofocusController.autoFocus(
                            rangez=autofocusRange,
                            resolutionz=autofocusResolution,
                            defocusz=0
                        )
                        self._logger.debug(f"Autofocus completed at position ({x}, {y})")
                    except Exception as e:
                        self._logger.error(f"Autofocus failed at position ({x}, {y}): {e}")

                numpy_array = self.mDetector.getLatestFrame()

                affine_matrix_four_d = [
                    [1, 0, 0, x],  # Please calcuate this correctly
                    [0, 1, 0, y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]

                mFrameList.append(self.mDetector.getLatestFrame())

                image = from_array_like(
                    xr.DataArray(
                        numpy_array,
                        dims=["y", "x"],
                    ),
                    transformation_views=[
                        PartialAffineTransformationViewInput(
                            affineMatrix=affine_matrix_four_d, stage=stage
                        )
                    ],
                    name=f"Tile Scan {x},{y}",
                )

                yield image

            # move back in x
            # mPositioner.moveTo(0, y, speed=speed, is_blocking=True) Deactived because virtual manager does not implemt that

        # mPositioner.moveTo(0, 0, speed=speed, is_blocking=True)

    @APIExport(runOnUIThread=False)
    def deconvolve(self) -> int:
        """Trigger deconvolution via Arkitekt."""
        # grab an image
        frame = self.mDetector.getLatestFrame()  # X,Y,C, uint8 numpy array
        numpy_array = list(frame)[0]

        # Deconvolve using Arkitekt
        deconvolved_image = self._master.arkitektManager.upload_and_deconvolve_image(
            numpy_array
        )
        # QUESTION: Is this a synchronous call? Do we need to wait for the result?
        # The result that came back was none

        if deconvolved_image is not None:
            print("Image deconvolution successful!")
            return 2
        else:
            print("Deconvolution failed, returning original image")
            return 1
