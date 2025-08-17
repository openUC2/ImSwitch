from qtpy import QtCore, QtWidgets
from imswitch.imcontrol.view import guitools
from .basewidgets import Widget


class ESP32InfoScreenWidget(Widget):
    """ Widget for ESP32 InfoScreen display integration """

    def __post_init__(self):
        # Main layout
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # Status display
        self.statusGroup = QtWidgets.QGroupBox('ESP32 InfoScreen Status')
        self.statusLayout = QtWidgets.QVBoxLayout()
        self.statusGroup.setLayout(self.statusLayout)
        
        self.connectionLabel = QtWidgets.QLabel('Connection: Disconnected')
        self.connectionLabel.setStyleSheet("QLabel { color: red; }")
        self.statusLayout.addWidget(self.connectionLabel)
        
        self.infoLabel = QtWidgets.QLabel('The ESP32 InfoScreen provides bidirectional control between\nthe ESP32 display and ImSwitch for:')
        self.statusLayout.addWidget(self.infoLabel)
        
        # Feature list
        features = [
            '• Motor/stage control via joystick',
            '• LED RGB matrix control',
            '• Objective slot switching (1, 2)', 
            '• Sample position visualization',
            '• Laser intensity control (PWM channels)',
            '• Image capture and display'
        ]
        
        for feature in features:
            featureLabel = QtWidgets.QLabel(feature)
            featureLabel.setStyleSheet("QLabel { margin-left: 10px; }")
            self.statusLayout.addWidget(featureLabel)
        
        self.layout.addWidget(self.statusGroup)
        
        # Control buttons
        self.controlGroup = QtWidgets.QGroupBox('Manual Controls')
        self.controlLayout = QtWidgets.QGridLayout()
        self.controlGroup.setLayout(self.controlLayout)
        
        # Connection control
        self.connectButton = guitools.BetterPushButton('Connect ESP32')
        self.connectButton.setCheckable(False)
        self.controlLayout.addWidget(self.connectButton, 0, 0)
        
        self.disconnectButton = guitools.BetterPushButton('Disconnect ESP32')  
        self.disconnectButton.setCheckable(False)
        self.controlLayout.addWidget(self.disconnectButton, 0, 1)
        
        # Test controls
        self.testLEDButton = guitools.BetterPushButton('Test LED (Red)')
        self.testLEDButton.setCheckable(False)
        self.controlLayout.addWidget(self.testLEDButton, 1, 0)
        
        self.sendTestImageButton = guitools.BetterPushButton('Send Test Image')
        self.sendTestImageButton.setCheckable(False)
        self.controlLayout.addWidget(self.sendTestImageButton, 1, 1)
        
        self.layout.addWidget(self.controlGroup)
        
        # Spacer to push everything to top
        self.layout.addStretch()
        
    def setConnectionStatus(self, connected: bool):
        """Update connection status display"""
        if connected:
            self.connectionLabel.setText('Connection: Connected')
            self.connectionLabel.setStyleSheet("QLabel { color: green; }")
            self.connectButton.setEnabled(False)
            self.disconnectButton.setEnabled(True)
            self.testLEDButton.setEnabled(True)
            self.sendTestImageButton.setEnabled(True)
        else:
            self.connectionLabel.setText('Connection: Disconnected')
            self.connectionLabel.setStyleSheet("QLabel { color: red; }")
            self.connectButton.setEnabled(True)
            self.disconnectButton.setEnabled(False)
            self.testLEDButton.setEnabled(False)
            self.sendTestImageButton.setEnabled(False)
            
    def replaceWithError(self, errorText):
        """Replace widget content with error message"""
        # Clear existing layout
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
        
        # Add error message
        errorLabel = QtWidgets.QLabel(f'Error: {errorText}')
        errorLabel.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        errorLabel.setWordWrap(True)
        self.layout.addWidget(errorLabel)
        
        # Add spacer
        self.layout.addStretch()