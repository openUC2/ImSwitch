import warnings

from .AutofocusManager import AutofocusManager
from .FOVLockManager import FOVLockManager
from .DetectorsManager import DetectorsManager, NoDetectorsError
from .LasersManager import LasersManager
from .LEDsManager import LEDsManager
from .LEDMatrixsManager import LEDMatrixsManager
from .MultiManager import MultiManager
from .NidaqManager import NidaqManager
from .PositionersManager import PositionersManager
from .RS232sManager import RS232sManager
from .RecordingManager import RecordingManager, RecMode, SaveMode, SaveFormat
from .SLMManager import SLMManager
from .ScanManagerPointScan import ScanManagerPointScan
from .ScanManagerBase import ScanManagerBase
from .ScanManagerMoNaLISA import ScanManagerMoNaLISA
from .StandManager import StandManager
from .RotatorsManager import RotatorsManager
try:
    from .UC2ConfigManager import UC2ConfigManager
except ModuleNotFoundError:
    warnings.warn("UC2ConfigManager not available; please install uc2rest module")
    # Create a mock UC2ConfigManager class so imports work
    UC2ConfigManager = type('UC2ConfigManager', (), {})
from .SIMManager import SIMManager
from .DPCManager import DPCManager
from .MCTManager import MCTManager
from .TimelapseManager import TimelapseManager
from .ExperimentManager import ExperimentManager
from .ROIScanManager import ROIScanManager
from .LightsheetManager import LightsheetManager
from .WebRTCManager import WebRTCManager
from .HyphaManager import HyphaManager
from .HistoScanManager import HistoScanManager
from .StresstestManager import StresstestManager
from .ObjectiveManager import ObjectiveManager
from .WorkflowManager import WorkflowManager
from .FlowStopManager import FlowStopManager
from .LepmonManager import LepmonManager
from .FlatfieldManager import FlatfieldManager
from .PixelCalibrationManager import PixelCalibrationManager
