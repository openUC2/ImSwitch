from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import dirtools, initLogger, APIExport

from arkitekt_next import register, easy, progress

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
        self._logger.debug('Initializing')
        
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        if len(allDetectorNames) == 0:
            return
        self.mDetector = self._master.detectorsManager[self._master.detectorsManager.getAllDeviceNames()[0]]


        self.arkitekt_app = self._master.arkitektManager.get_arkitekt_app()
        self.arkitekt_app.register(self.moveToSampleLoadingPosition)
        self.arkitekt_app.register(self.runTileScan)
        self.arkitekt_app.run_detached()

    def moveToSampleLoadingPosition(self, speed:float=10000, is_blocking:bool=True):
        """ Move to sample loading position. """
        positionerNames = self._master.positionersManager.getAllDeviceNames()
        if len(positionerNames) == 0:
            self._logger.warning("No positioners available to move to sample loading position.")
            return
        positionerName = positionerNames[0]
        self._logger.debug(f"Moving to sample loading position for positioner {positionerName}")
        self._master.positionersManager[positionerName].moveToSampleLoadingPosition(speed=speed, is_blocking=is_blocking)

    def runTileScan(self, positionerName:str=None, xRange:float=100, yRange:float=100, xStep:int=10, yStep:int=10, speed:float=10000):
        """ Run a tile scan. """
        # TODO: Implement a check if the positioner supports tile scanning
        if positionerName is None:
            positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self._logger.debug(f"Starting tile scan for positioner {positionerName}")
        # have a tile scan function in the positioner manager inside a for loop
        mFrameList = []
        mPositioner = self._master.positionersManager[positionerName]
        for y in range(0, yRange, yStep):
            for x in range(0, xRange, xStep):
                mPositioner.move(value=(x,y), axis="XY", is_absolute=True)
                mFrameList.append(self.mDetector.getLatestFrame())
            # move back in x
            mPositioner.moveTo(0, y, speed=speed, is_blocking=True)

        mPositioner.moveTo(0, 0, speed=speed, is_blocking=True)
        return mFrameList
    
    @APIExport(runOnUIThread=False)
    def deconvolve(self) -> int:
        """Trigger deconvolution via Arkitekt."""
        # grab an image
        frame = self.mDetector.getLatestFrame()  # X,Y,C, uint8 numpy array
        numpy_array = list(frame)[0]
                
        # Deconvolve using Arkitekt
        deconvolved_image = self._master.arkitektManager.upload_and_deconvolve_image(numpy_array)
        # QUESTION: Is this a synchronous call? Do we need to wait for the result? 
        # The result that came back was none
        
        if deconvolved_image is not None:
            print("Image deconvolution successful!")
            return 2
        else:
            print("Deconvolution failed, returning original image")
            return 1
