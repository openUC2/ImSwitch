from ..basecontrollers import ImConWidgetController
from imswitch.imcommon.model import dirtools, initLogger, APIExport
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
