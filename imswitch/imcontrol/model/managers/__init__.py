import warnings
# TODO: Import managers dynamically - similar to the controllers - to save time
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
# RecordingManager is deprecated - use RecordingService from io module
# Importing RecMode, SaveMode, SaveFormat from io module for backwards compatibility
from imswitch.imcontrol.model.io import RecordingService, RecMode, SaveMode, SaveFormat
# Create alias for backwards compatibility (will be removed in future versions)
RecordingManager = RecordingService
from .SLMManager import SLMManager
from .ScanManagerPointScan import ScanManagerPointScan
from .ScanManagerBase import ScanManagerBase
from .RotatorsManager import RotatorsManager
from .UC2ConfigManager import UC2ConfigManager
from .SIMManager import SIMManager
from .DPCManager import DPCManager
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
from .ArkitektManager import ArkitektManager
from .InstrumentMetadataManager import InstrumentMetadataManager
