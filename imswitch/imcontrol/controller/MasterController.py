from imswitch.imcommon.model import VFileItem, initLogger

import pkg_resources

from imswitch.imcontrol.model import (
    DetectorsManager, LasersManager, MultiManager, PositionersManager,
    RecordingManager, RS232sManager, SLMManager, SIMManager, DPCManager, LEDMatrixsManager, MCTManager, ROIScanManager, WebRTCManager, HyphaManager,
    UC2ConfigManager, AutofocusManager, HistoScanManager, StresstestManager, PixelCalibrationManager, LightsheetManager, NidaqManager, FOVLockManager,
    StandManager, RotatorsManager, LEDsManager, ScanManagerBase, ScanManagerPointScan, ScanManagerMoNaLISA, FlatfieldManager, 
    FlowStopManager, WorkflowManager, TimelapseManager, LepmonManager, ExperimentManager, ObjectiveManager
)


class MasterController:
    """
    This class will handle the communication between software and hardware,
    using the managers for each hardware set.
    """

    def __init__(self, setupInfo, commChannel, moduleCommChannel):
        self.__logger = initLogger(self)
        self.__setupInfo = setupInfo
        self.__commChannel = commChannel
        self.__moduleCommChannel = moduleCommChannel

        # Init managers
        self.rs232sManager = RS232sManager(self.__setupInfo.rs232devices)

        lowLevelManagers = {
            'rs232sManager': self.rs232sManager
        }

        self.detectorsManager = DetectorsManager(self.__setupInfo.detectors, updatePeriod=100,
                                                 **lowLevelManagers)
        self.lasersManager = LasersManager(self.__setupInfo.lasers,
                                           **lowLevelManagers)
        self.positionersManager = PositionersManager(self.__setupInfo.positioners,
                                                     self.__commChannel,
                                                     **lowLevelManagers)
        self.LEDMatrixsManager = LEDMatrixsManager(self.__setupInfo.LEDMatrixs,
                                           **lowLevelManagers)
        self.rotatorsManager = RotatorsManager(self.__setupInfo.rotators,
                                            **lowLevelManagers)

        self.LEDsManager = LEDsManager(self.__setupInfo.LEDs)
        #self.scanManager = ScanManager(self.__setupInfo)
        self.recordingManager = RecordingManager(self.detectorsManager)
        if "SLM" in self.__setupInfo.availableWidgets: self.slmManager = SLMManager(self.__setupInfo.slm)
        self.UC2ConfigManager = UC2ConfigManager(self.__setupInfo.uc2Config, lowLevelManagers)
        if "SIM" in self.__setupInfo.availableWidgets: self.simManager = SIMManager(self.__setupInfo.sim)
        if "DPC" in self.__setupInfo.availableWidgets: self.dpcManager = DPCManager(self.__setupInfo.dpc)
        if "MCT" in self.__setupInfo.availableWidgets: self.mctManager = MCTManager(self.__setupInfo.mct)
        self.nidaqManager = NidaqManager(self.__setupInfo.nidaq)
        self.roiscanManager = ROIScanManager(self.__setupInfo.roiscan)
        if "Lightsheet" in self.__setupInfo.availableWidgets: self.lightsheetManager = LightsheetManager(self.__setupInfo.lightsheet)
        if "WebRTC" in self.__setupInfo.availableWidgets: self.webrtcManager = WebRTCManager(self.__setupInfo.webrtc)
        if "Timelapse" in self.__setupInfo.availableWidgets: self.timelapseManager = TimelapseManager()
        if "Experiment" in self.__setupInfo.availableWidgets: self.experimentManager = ExperimentManager()
        if "Objective" in self.__setupInfo.availableWidgets: self.objectiveManager = ObjectiveManager(self.__setupInfo.objective)
        if "HistoScan" in self.__setupInfo.availableWidgets: self.HistoScanManager = HistoScanManager(self.__setupInfo.HistoScan)
        if "Stresstest" in self.__setupInfo.availableWidgets: self.StresstestManager = StresstestManager(self.__setupInfo.Stresstest)
        if "FlowStop" in self.__setupInfo.availableWidgets: self.FlowStopManager = FlowStopManager(self.__setupInfo.FlowStop)
        if "Lepmon" in self.__setupInfo.availableWidgets: self.LepmonManager = LepmonManager(self.__setupInfo.Lepmon)
        if "FlatField" in self.__setupInfo.availableWidgets: self.FlatfieldManager = FlatfieldManager(self.__setupInfo.Flatfield)
        if "PixelCalibration" in self.__setupInfo.availableWidgets: self.PixelCalibrationManager = PixelCalibrationManager(self.__setupInfo.PixelCalibration)
        if "AutoFocus" in self.__setupInfo.availableWidgets: self.AutoFocusManager = AutofocusManager(self.__setupInfo.autofocus)
        if "FOV" in self.__setupInfo.availableWidgets: self.FOVLockManager = FOVLockManager(self.__setupInfo.fovLock)
        if "Workflow" in self.__setupInfo.availableWidgets: self.workflowManager = WorkflowManager()
        # load all implugin-related managers and add them to the class
        # try to get it from the plugins
        # If there is a imswitch_sim_manager, we want to add this as self.imswitch_sim_widget to the
        # MasterController Class

        for entry_point in pkg_resources.iter_entry_points(f'imswitch.implugins'):
            InfoClass = None
            print (f"entry_point: {entry_point.name}")
            try:
                if entry_point.name.find("manager")>=0:
                    # check if there is an info class, too
                    try:
                        InfoClassName = entry_point.name.split("_manager")[0] + "_info"
                        # load the info class from InfoClassName
                        InfoClass = pkg_resources.load_entry_point("imswitch", "imswitch.implugins", InfoClassName)
                    except Exception as e:
                        InfoClass = None
                    ManagerClass = entry_point.load(InfoClass)  # Load the manager class
                    # self.__setupInfo.add_attribute(attr_name=entry_point.name.split("_manager")[0], attr_value={})
                    moduleInfo = None # TODO: This is not complete yet - the setupinfo would need to be added to the class in the very begnning prior to detecing external plugins/hooks
                    manager = ManagerClass(moduleInfo)  # Initialize the manager
                    setattr(self, entry_point.name, manager)  # Add the manager to the class
            except Exception as e:
                self.__logger.error(e)

        if self.__setupInfo.microscopeStand:
            self.standManager = StandManager(self.__setupInfo.microscopeStand,
                                             **lowLevelManagers)

        # Generate scanManager type according to setupInfo
        if self.__setupInfo.scan:
            if self.__setupInfo.scan.scanWidgetType == "PointScan":
                self.scanManager = ScanManagerPointScan(self.__setupInfo)
            elif self.__setupInfo.scan.scanWidgetType == "Base":
                self.scanManager = ScanManagerBase(self.__setupInfo)
            elif self.__setupInfo.scan.scanWidgetType == "MoNaLISA":
                self.scanManager = ScanManagerMoNaLISA(self.__setupInfo)
            else:
                self.__logger.error(
                    'ScanWidgetType in SetupInfo["scan"] not recognized, choose one of the following:'
                    ' ["Base", "PointScan", "MoNaLISA"].'
                )
                return

        # Connect signals
        cc = self.__commChannel

        self.detectorsManager.sigAcquisitionStarted.connect(cc.sigAcquisitionStarted)
        self.detectorsManager.sigAcquisitionStopped.connect(cc.sigAcquisitionStopped)
        self.detectorsManager.sigDetectorSwitched.connect(cc.sigDetectorSwitched)
        self.detectorsManager.sigImageUpdated.connect(cc.sigUpdateImage)
        self.detectorsManager.sigNewFrame.connect(cc.sigNewFrame)

        self.recordingManager.sigRecordingStarted.connect(cc.sigRecordingStarted)
        self.recordingManager.sigRecordingEnded.connect(cc.sigRecordingEnded)
        self.recordingManager.sigRecordingFrameNumUpdated.connect(cc.sigUpdateRecFrameNum)
        self.recordingManager.sigRecordingTimeUpdated.connect(cc.sigUpdateRecTime)
        self.recordingManager.sigMemorySnapAvailable.connect(cc.sigMemorySnapAvailable)
        self.recordingManager.sigMemoryRecordingAvailable.connect(self.memoryRecordingAvailable)

        if "SLM" in self.__setupInfo.availableWidgets:
            self.slmManager.sigSLMMaskUpdated.connect(cc.sigSLMMaskUpdated)
            self.simManager.sigSIMMaskUpdated.connect(cc.sigSIMMaskUpdated)

    def memoryRecordingAvailable(self, name, file, filePath, savedToDisk):
        self.__moduleCommChannel.memoryRecordings[name] = VFileItem(
            data=file, filePath=filePath, savedToDisk=savedToDisk
        )

    def closeEvent(self):
        self.recordingManager.endRecording(emitSignal=False, wait=True)

        for attrName in dir(self):
            attr = getattr(self, attrName)
            if isinstance(attr, MultiManager):
                attr.finalize()


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
