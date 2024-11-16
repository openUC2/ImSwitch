
from imswitch.imcommon.model import dirtools, modulesconfigtools, ostools, APIExport
from imswitch.imcommon.framework import Signal, Worker, Mutex, Timer
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.controller.basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS



class ObjectiveController(LiveUpdatedController):
    """ Linked to ObjectiveWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)

        # connect camera and stage - 
        self.positionerName = self._master.positionersManager.getAllDeviceNames()[0]
        self.positioner = self._master.positionersManager[self.positionerName]
        self.detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
        self.detector = self._master.detectorsManager[self.detectorName]
        
        # Define the parameters for the objective revolver
        self.currentObjective = 0
        
        #TODO this should be loaded from manager
        self.revolverAxis = "A"
        self.posObjective1 = 0
        self.posObjective2 = 20000
        self.nameObjective1 = 10
        self.nameObjective2 = 20
        self.magnObjective1 = "10x"
        self.magnObjective2 = "20x"
        self.pEffObjective1 = .2
        self.pEffObjective2 = .1
        self.calibrateOnStart = True
        
        # initialize Objective Store
        self.objectiveStorer = ObjecitveStorer()
        self.objectiveStorer.setObjectiveByID(0, position=self.posObjective1, name=self.nameObjective1, magnification=self.magnObjective1, pixelsize_eff=self.pEffObjective1)
        self.objectiveStorer.setObjectiveByID(1, position=self.posObjective2, name=self.nameObjective2, magnification=self.magnObjective2, pixelsize_eff=self.pEffObjective2)
        
        # initialize objective lens mover
        self.objectiveLensMover = ObjectiveLensMover(positioner = self.positioner, positionerName = self.positionerName, 
                                                     revolverAxis = self.revolverAxis, objectiveStorer = self.objectiveStorer)

        if self.calibrateOnStart:
            self.calibrateObjective()
            
        if IS_HEADLESS:
            return

        # Connect signals to slots
        self._widget.btnObj1.clicked.connect(self.onObj1Clicked)
        self._widget.btnObj2.clicked.connect(self.onObj2Clicked)
        self._widget.btnCalibrate.clicked.connect(self.onCalibrateClicked)
        self._widget.btnMovePlus.clicked.connect(self.onMovePlusClicked)
        self._widget.btnMoveMinus.clicked.connect(self.onMoveMinusClicked)
        self._widget.btnSetPosObj1.clicked.connect(self.onSetPosObj1Clicked)
        #self._widget.btnSetPosObj2.clicked.connect(self.onSetPosObj2Clicked)
        
        
    @APIExport(runOnUIThread=True)
    def calibrateObjective(self):
        '''This homes the objective revolver.'''
        self.objectiveLensMover.calibrateObjective()
        self.moveToObjectiveID(self.currentObjective)
        if not IS_HEADLESS: self._widget.setCurrentObjectiveInfo(self.currentObjective)
        
    @APIExport(runOnUIThread=True)
    def moveToObjectiveID(self, objectiveID: int):
        '''This moves the objective revolver to the objectiveID.'''
        self.currentObjective = objectiveID
        self.objectiveLensMover.moveToObjectiveID(objectiveID)
        # update the pixelsize_eff
        self.detector.setPixelSizeUm(self.objectiveStorer.getObjectiveByID(objectiveID).pixelsize_eff)
        # self._master._MasterController__commChannel.sigPixelSizeChange.emit()
        if not IS_HEADLESS: self._widget.setCurrentObjectiveInfo(self.currentObjective)
        
    @APIExport(runOnUIThread=True)
    def getCurrentObjective(self):
        return self.currentObjective, self.objectiveStorer.getObjectiveByID(self.currentObjective).name

    def onObj1Clicked(self):
        # move to objective 1
        self.moveToObjectiveID(1)

    def onObj2Clicked(self):
        # move to objective 2
        self.moveToObjectiveID(2)

    def onCalibrateClicked(self):
        # move to objective 1 after homing
        self.calibrateObjective()

    def onMovePlusClicked(self):
        # Define what happens when Move + is clicked
        self.positioner.move(value=self.incrementPosition, speed=self.speed, axis=self.revolverAxis, is_absolute=False, is_blocking=False)  

    def onMoveMinusClicked(self):
        # Define what happens when Move - is clicked
        self.positioner.move(value=-self.incrementPosition, speed=self.speed, axis=self.revolverAxis, is_absolute=False, is_blocking=False)

    def onSetPosObj1Clicked(self):
        # Define what happens when Set Position Objective 1 is clicked
        self.positioner.setPositionOnDevice(axis=self.revolverAxis, value=0)
        self.currentObjective = 1
        
    def onSetPosObj2Clicked(self):
        # Define what happens when Set Position Objective 2 is clicked
        self.positioner.setPositionOnDevice(axis=self.revolverAxis, value=self.posObjective2)
        self.currentObjective = 2

class Objective(object):
    '''This class stores the information of an objective lens.'''
    position = 0
    name = ""
    magnification = 0
    pixelsize_eff = 0
    
    def __init__(self, position, name, magnification, pixelsize_eff):
        self.position = position
        self.name = name
        self.magnification = magnification
        self.pixelsize_eff = pixelsize_eff    

class ObjecitveStorer:
    '''This class stores the information of all objective lenses.
        We can store hold up to 2 objective lenses.'''
    def __init__(self):
        self.objectives = []
        self.objectives.append(Objective(0, "Objective 1", 0, 0))
        self.objectives.append(Objective(0, "Objective 2", 0, 0))
        
    def setObjectiveByID(self, id, position, name, magnification, pixelsize_eff):
        self.objectives[id] = Objective(position, name, magnification, pixelsize_eff)
        
    def getObjectiveByID(self, id):
        return self.objectives[id]
    
class ObjectiveLensMover(object):
    
    '''This class organizes the motion of the objective lens revolver.'''
    maxDistance = 30000
    offsetFromZero = 1000 
    speed = 25000
    incrementPosition = 50
    currentObjective = -1
    
    def __init__(self, positioner, positionerName, revolverAxis, objectiveStorer):
        self.positioner = positioner
        self.positionerName = positionerName
        self.revolverAxis = revolverAxis
        self.objectiveStorer = objectiveStorer
        
    def calibrateObjective(self):
        '''This function calibrates the objective revolver by moving to the first objective lens.'''
        # Move the revolver to the most negative position and then to offsetFromZero in opposite direction e.g. homing
        # TODO: This has to be reworked!! 
        self.positioner.move(value=-1.5*(self.offsetFromZero+self.objectiveStorer.getObjectiveByID(0).position), speed=self.speed, axis=self.revolverAxis, is_absolute=False, is_blocking=False)
        self.positioner.move(value=self.offsetFromZero, speed=self.speed, axis=self.revolverAxis, is_absolute=False, is_blocking=False)
        self.positioner.setPositionOnDevice(axis=self.revolverAxis, value=0)
        self.currentObjective = 1
        #if self._widget.setCurrentObjectiveInfo(self.currentObjective)
        
    def moveToObjectiveID(self, objectiveID):
        # Move the revolver to the objectiveID
        mPos = self.objectiveStorer.getObjectiveByID(objectiveID).position
        self.positioner.move(value=mPos, speed=self.speed, axis=self.revolverAxis, is_absolute=True, is_blocking=False)
        self.currentObjective = objectiveID

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
