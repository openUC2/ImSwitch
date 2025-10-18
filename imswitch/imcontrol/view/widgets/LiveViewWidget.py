"""
LiveViewWidget - Minimal widget for LiveViewController.

This widget provides a simple UI for managing live streaming when not in headless mode.
Most functionality is accessed via API in headless mode.
"""

from qtpy import QtCore, QtWidgets
from imswitch.imcontrol.view import guitools
from .basewidgets import Widget


class LiveViewWidget(Widget):
    """Widget for live streaming control."""
    
    sigStartStream = QtCore.Signal(str, str)  # (detectorName, protocol)
    sigStopStream = QtCore.Signal(str, str)   # (detectorName, protocol)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Protocol selection
        self.protocolLabel = QtWidgets.QLabel('Protocol:')
        self.protocolCombo = QtWidgets.QComboBox()
        self.protocolCombo.addItems(['binary', 'jpeg', 'mjpeg', 'webrtc'])
        
        # Detector selection (will be populated by controller)
        self.detectorLabel = QtWidgets.QLabel('Detector:')
        self.detectorCombo = QtWidgets.QComboBox()
        
        # Control buttons
        self.startButton = guitools.BetterPushButton('Start Stream')
        self.stopButton = guitools.BetterPushButton('Stop Stream')
        self.stopButton.setEnabled(False)
        
        # Status display
        self.statusLabel = QtWidgets.QLabel('Status:')
        self.statusText = QtWidgets.QLineEdit('Idle')
        self.statusText.setReadOnly(True)
        
        # Active streams display
        self.activeStreamsLabel = QtWidgets.QLabel('Active Streams:')
        self.activeStreamsList = QtWidgets.QListWidget()
        self.activeStreamsList.setMaximumHeight(100)
        
        # Layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        
        # Protocol and detector selection
        selectionLayout = QtWidgets.QGridLayout()
        selectionLayout.addWidget(self.protocolLabel, 0, 0)
        selectionLayout.addWidget(self.protocolCombo, 0, 1)
        selectionLayout.addWidget(self.detectorLabel, 1, 0)
        selectionLayout.addWidget(self.detectorCombo, 1, 1)
        layout.addLayout(selectionLayout)
        
        # Control buttons
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.startButton)
        buttonLayout.addWidget(self.stopButton)
        layout.addWidget(QtWidgets.QWidget())  # Spacer
        layout.addLayout(buttonLayout)
        
        # Status
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.statusText)
        
        # Active streams
        layout.addWidget(self.activeStreamsLabel)
        layout.addWidget(self.activeStreamsList)
        
        layout.addStretch()
        
        # Connect signals
        self.startButton.clicked.connect(self._onStartClicked)
        self.stopButton.clicked.connect(self._onStopClicked)
    
    def _onStartClicked(self):
        """Handle start button click."""
        detector = self.detectorCombo.currentText()
        protocol = self.protocolCombo.currentText()
        self.sigStartStream.emit(detector, protocol)
    
    def _onStopClicked(self):
        """Handle stop button click."""
        detector = self.detectorCombo.currentText()
        protocol = self.protocolCombo.currentText()
        self.sigStopStream.emit(detector, protocol)
    
    def setDetectorList(self, detectors):
        """Set the list of available detectors."""
        self.detectorCombo.clear()
        self.detectorCombo.addItems(detectors)
    
    def setStatus(self, status):
        """Set status text."""
        self.statusText.setText(status)
    
    def updateActiveStreams(self, streams):
        """Update the list of active streams."""
        self.activeStreamsList.clear()
        for stream in streams:
            detector = stream.get('detector', 'Unknown')
            protocol = stream.get('protocol', 'Unknown')
            self.activeStreamsList.addItem(f"{detector} - {protocol}")
    
    def setButtonsEnabled(self, start_enabled, stop_enabled):
        """Enable/disable control buttons."""
        self.startButton.setEnabled(start_enabled)
        self.stopButton.setEnabled(stop_enabled)


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
