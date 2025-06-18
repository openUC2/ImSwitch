import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

# microeye gui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
try:
    from microEye.fitting.fit import BlobDetectionWidget, DoG_FilterWidget, BandpassFilterWidget, TemporalMedianFilterWidget
    from microEye.fitting.results import FittingMethod, FittingResults
    from microEye.checklist_dialog import ChecklistDialog, Checklist
    isMicroEye = True
except:
    isMicroEye = False

from imswitch.imcommon.view.guitools import pyqtgraphtools
from imswitch.imcontrol.view import guitools
from .basewidgets import NapariHybridWidget


class STORMReconWidget(NapariHybridWidget):
    """ Displays the STORMRecon transform of the image. """

    sigShowToggled = QtCore.Signal(bool)  # (enabled)
    sigUpdateRateChanged = QtCore.Signal(float)  # (rate)
    sigSliderValueChanged = QtCore.Signal(float)  # (value)
    
    # New signals for backend API functionality
    sigStartFastAcquisition = QtCore.Signal(dict)  # (acquisition_params)
    sigStopFastAcquisition = QtCore.Signal()
    sigGetFrameGenerator = QtCore.Signal(int, float)  # (num_frames, timeout)
    sigGetStatus = QtCore.Signal()
    sigSetParameters = QtCore.Signal(dict)  # (parameters)
    sigTriggerReconstruction = QtCore.Signal()

    def __post_init__(self):

        # Napari image layer for results
        self.layer = None

        # Main GUI
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        # Side TabView
        self.tabView = QTabWidget()
        self.layout.addWidget(self.tabView, 0)

        if not isMicroEye:
            self.image_control_layout = QFormLayout()
            self.image_control_layout.addRow(
                QLabel('MicroEye not installed. Please install MicroEye to use this feature.'),
                None)
            
            # Still add the Fast Acquisition API tab even without microEye
            self.fast_acq_group = QWidget()
            self.fast_acq_layout = QVBoxLayout()
            self.fast_acq_group.setLayout(self.fast_acq_layout)
            self.tabView.addTab(self.fast_acq_group, 'Fast Acquisition API')
            
            self._setupFastAcquisitionTab()
            return


        # Localization / Render tab layout
        self.loc_group = QWidget()
        self.loc_form = QFormLayout()
        self.loc_group.setLayout(self.loc_form)

        # results stats tab layout
        self.data_filters = QWidget()
        self.data_filters_layout = QVBoxLayout()
        self.data_filters.setLayout(self.data_filters_layout)

        # Tiff Options tab layout
        self.controls_group = QWidget()
        self.controls_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_layout)

        self.image_control_layout = QFormLayout()

        # Add tabView
        self.tabView.addTab(self.controls_group, 'Prefit Options')
        self.tabView.addTab(self.loc_group, 'Fitting')
        self.tabView.addTab(self.data_filters, 'Data Filters')
        
        # Add new Fast Acquisition API tab
        self.fast_acq_group = QWidget()
        self.fast_acq_layout = QVBoxLayout()
        self.fast_acq_group.setLayout(self.fast_acq_layout)
        self.tabView.addTab(self.fast_acq_group, 'Fast Acquisition API')
        
        self._setupFastAcquisitionTab()

        #self.detection = QCheckBox('Enable Realtime localization.')
        #self.detection.setChecked(False)

        #self.saveCropped = QPushButton(
        #    'Save Cropped Image',
        #    clicked=lambda: self.save_cropped_img())

        #self.image_control_layout.addWidget(self.detection)
        #self.image_control_layout.addWidget(self.saveCropped)

        '''
        PREFIT OPTIONS
        '''

        self.controls_layout.addLayout(
            self.image_control_layout)

        self.blobDetectionWidget = BlobDetectionWidget()
        #self.blobDetectionWidget.update.connect(
        #    lambda: self.update_display())

        self.detection_method = QComboBox()
        # self.detection_method.currentIndexChanged.connect()
        self.detection_method.addItem(
            'OpenCV Blob Detection',
            self.blobDetectionWidget
        )

        self.doG_FilterWidget = DoG_FilterWidget()
        self.doG_FilterWidget.update.connect(
            lambda: self.update_display())
        self.bandpassFilterWidget = BandpassFilterWidget()
        self.bandpassFilterWidget.setVisible(False)
        #self.bandpassFilterWidget.update.connect(
        #    lambda: self.update_display())

        self.image_filter = QComboBox()
        self.image_filter.addItem(
            'Difference of Gaussians',
            self.doG_FilterWidget)
        self.image_filter.addItem(
            'Fourier Bandpass Filter',
            self.bandpassFilterWidget)

        # displays the selected item
        def update_visibility(box: QComboBox):
            for idx in range(box.count()):
                box.itemData(idx).setVisible(
                    idx == box.currentIndex())

        self.detection_method.currentIndexChanged.connect(
            lambda: update_visibility(self.detection_method))
        self.image_filter.currentIndexChanged.connect(
            lambda: update_visibility(self.image_filter))

        self.image_control_layout.addRow(
            QLabel('Approx. Loc. Method:'),
            self.detection_method)
        self.image_control_layout.addRow(
            QLabel('Image filter:'),
            self.image_filter)

        self.th_min_label = QLabel('Relative threshold:')
        self.th_min_slider = QDoubleSpinBox()
        self.th_min_slider.setMinimum(0)
        self.th_min_slider.setMaximum(100)
        self.th_min_slider.setSingleStep(0.01)
        self.th_min_slider.setDecimals(3)
        self.th_min_slider.setValue(0.2)
        #self.th_min_slider.valueChanged.connect(self.slider_changed)

        self.image_control_layout.addRow(
            self.th_min_label,
            self.th_min_slider)

        self.tempMedianFilter = TemporalMedianFilterWidget()
        #self.tempMedianFilter.update.connect(lambda: self.update_display())
        #self.controls_layout.addWidget(self.tempMedianFilter)

        self.controls_layout.addWidget(self.blobDetectionWidget)
        self.controls_layout.addWidget(self.doG_FilterWidget)
        self.controls_layout.addWidget(self.bandpassFilterWidget)

        #self.pages_slider.valueChanged.connect(self.slider_changed)
        #self.min_slider.valueChanged.connect(self.slider_changed)
        #self.max_slider.valueChanged.connect(self.slider_changed)
        #self.autostretch.stateChanged.connect(self.slider_changed)
        #self.detection.stateChanged.connect(self.slider_changed)

        self.controls_layout.addStretch()




        '''
        FITTING
        '''

        # Localization / Render layout
        self.fitting_cbox = QComboBox()
        self.fitting_cbox.addItem(
            '2D Phasor-Fit (CPU)',
            FittingMethod._2D_Phasor_CPU)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit fixed sigma (GPU/CPU)',
            FittingMethod._2D_Gauss_MLE_fixed_sigma)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit free sigma (GPU/CPU)',
            FittingMethod._2D_Gauss_MLE_free_sigma)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit elliptical sigma (GPU/CPU)',
            FittingMethod._2D_Gauss_MLE_elliptical_sigma)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit cspline (GPU/CPU)',
            FittingMethod._3D_Gauss_MLE_cspline_sigma)

        self.render_cbox = QComboBox()
        self.render_cbox.addItem('2D Histogram', 0)
        self.render_cbox.addItem('2D Gaussian Histogram', 1)

        self.frc_cbox = QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Check Pattern')

        self.export_options = Checklist(
                'Exported Columns',
                ['Super-res image', ] + FittingResults.uniqueKeys(None),
                checked=True)

        self.export_precision = QLineEdit('%10.5f')

        self.px_size = QDoubleSpinBox()
        self.px_size.setMinimum(0)
        self.px_size.setMaximum(20000)
        self.px_size.setValue(117.5)

        self.super_px_size = QSpinBox()
        self.super_px_size.setMinimum(0)
        self.super_px_size.setMaximum(200)
        self.super_px_size.setValue(10)

        self.fit_roi_size = QSpinBox()
        self.fit_roi_size.setMinimum(7)
        self.fit_roi_size.setMaximum(99)
        self.fit_roi_size.setSingleStep(2)
        self.fit_roi_size.lineEdit().setReadOnly(True)
        self.fit_roi_size.setValue(13)

        self.drift_cross_args = QHBoxLayout()
        self.drift_cross_bins = QSpinBox()
        self.drift_cross_bins.setValue(10)
        self.drift_cross_px = QSpinBox()
        self.drift_cross_px.setValue(10)
        self.drift_cross_up = QSpinBox()
        self.drift_cross_up.setMaximum(1000)
        self.drift_cross_up.setValue(100)
        self.drift_cross_args.addWidget(self.drift_cross_bins)
        self.drift_cross_args.addWidget(self.drift_cross_px)
        self.drift_cross_args.addWidget(self.drift_cross_up)

        self.nneigh_merge_args = QHBoxLayout()
        self.nn_neighbors = QSpinBox()
        self.nn_neighbors.setValue(1)
        self.nn_min_distance = QDoubleSpinBox()
        self.nn_min_distance.setMaximum(20000)
        self.nn_min_distance.setValue(0)
        self.nn_max_distance = QDoubleSpinBox()
        self.nn_max_distance.setMaximum(20000)
        self.nn_max_distance.setValue(30)
        self.nn_max_off = QSpinBox()
        self.nn_max_off.setValue(1)
        self.nn_max_length = QSpinBox()
        self.nn_max_length.setMaximum(20000)
        self.nn_max_length.setValue(500)
        self.nneigh_merge_args.addWidget(self.nn_neighbors)
        self.nneigh_merge_args.addWidget(self.nn_min_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_off)
        self.nneigh_merge_args.addWidget(self.nn_max_length)

        self.loc_btn = QPushButton(
            'Localize',
            clicked=lambda: self.localize())
        self.refresh_btn = QPushButton(
            'Refresh SuperRes Image',
            clicked=lambda: self.renderLoc())

        self.frc_res_btn = QPushButton(
            'FRC Resolution',
            clicked=lambda: self.FRC_estimate())
        self.drift_cross_btn = QPushButton(
            'Drift cross-correlation',
            clicked=lambda: self.drift_cross())

        self.nn_layout = QHBoxLayout()
        self.nneigh_btn = QPushButton(
            'Nearest-neighbour',
            clicked=lambda: self.nneigh())
        self.merge_btn = QPushButton(
            'Merge Tracks',
            clicked=lambda: self.merge())
        self.nneigh_merge_btn = QPushButton(
            'NM + Merging',
            clicked=lambda: self.nneigh_merge())

        self.drift_fdm_btn = QPushButton(
            'Fiducial marker drift correction',
            clicked=lambda: self.drift_fdm())

        self.nn_layout.addWidget(self.nneigh_btn)
        self.nn_layout.addWidget(self.merge_btn)
        self.nn_layout.addWidget(self.nneigh_merge_btn)

        self.im_exp_layout = QHBoxLayout()
        self.import_loc_btn = QPushButton(
            'Import',
            clicked=lambda: self.import_loc())
        self.export_loc_btn = QPushButton(
            'Export',
            clicked=lambda: self.export_loc())

        self.im_exp_layout.addWidget(self.import_loc_btn)
        self.im_exp_layout.addWidget(self.export_loc_btn)

        self.loc_form.addRow(
            QLabel('Fitting:'),
            self.fitting_cbox
        )
        self.loc_form.addRow(
            QLabel('Rendering Method:'),
            self.render_cbox
        )
        self.loc_form.addRow(
            QLabel('Fitting roi-size [pixel]:'),
            self.fit_roi_size
        )
        self.loc_form.addRow(
            QLabel('Pixel-size [nm]:'),
            self.px_size
        )
        self.loc_form.addRow(
            QLabel('S-res pixel-size [nm]:'),
            self.super_px_size
        )

        # activate live localization
        self.loc_ref_lay = QHBoxLayout()
        self.loc_ref_lay.addWidget(self.loc_btn)
        self.loc_ref_lay.addWidget(self.refresh_btn)
        self.loc_form.addRow(self.loc_ref_lay)

        #self.loc_form.addRow(self.frc_res_btn) # don't need this button for now

        self.loc_form.addRow(
            QLabel('Drift X-Corr. (bins, pixelSize, upsampling):'))
        self.loc_form.addRow(self.drift_cross_args)
        self.loc_form.addRow(self.drift_cross_btn)
        self.loc_form.addRow(
            QLabel('NN (n-neighbor, min, max-distance, max-off, max-len):'))
        self.loc_form.addRow(self.nneigh_merge_args)
        self.loc_form.addRow(self.nn_layout)
        self.loc_form.addRow(self.drift_fdm_btn)
        self.loc_form.addRow(self.export_options)
        self.loc_form.addRow(
            QLabel('Format:'),
            self.export_precision)
        self.loc_form.addRow(self.im_exp_layout)


    def _setupFastAcquisitionTab(self):
        """Setup the Fast Acquisition API tab with backend controls"""
        
        # Main form layout
        fast_acq_form = QFormLayout()
        
        # Session Management Section
        session_group = QGroupBox("Session Management")
        session_layout = QFormLayout()
        
        self.session_id_input = QLineEdit()
        self.session_id_input.setPlaceholderText("experiment_001")
        session_layout.addRow(QLabel("Session ID:"), self.session_id_input)
        
        session_group.setLayout(session_layout)
        self.fast_acq_layout.addWidget(session_group)
        
        # Cropping Parameters Section
        crop_group = QGroupBox("Cropping Parameters")
        crop_layout = QFormLayout()
        
        self.crop_enabled = QCheckBox("Enable Cropping")
        crop_layout.addRow(self.crop_enabled)
        
        self.crop_x = QSpinBox()
        self.crop_x.setMaximum(10000)
        self.crop_x.setValue(100)
        crop_layout.addRow(QLabel("Crop X:"), self.crop_x)
        
        self.crop_y = QSpinBox()
        self.crop_y.setMaximum(10000)
        self.crop_y.setValue(100)
        crop_layout.addRow(QLabel("Crop Y:"), self.crop_y)
        
        self.crop_width = QSpinBox()
        self.crop_width.setMaximum(10000)
        self.crop_width.setValue(512)
        crop_layout.addRow(QLabel("Crop Width:"), self.crop_width)
        
        self.crop_height = QSpinBox()
        self.crop_height.setMaximum(10000)
        self.crop_height.setValue(512)
        crop_layout.addRow(QLabel("Crop Height:"), self.crop_height)
        
        crop_group.setLayout(crop_layout)
        self.fast_acq_layout.addWidget(crop_group)
        
        # Saving Options Section
        save_group = QGroupBox("Saving Options")
        save_layout = QFormLayout()
        
        self.save_enabled = QCheckBox("Enable Saving")
        save_layout.addRow(self.save_enabled)
        
        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("/data/storm_session.zarr")
        save_layout.addRow(QLabel("Save Path:"), self.save_path_input)
        
        self.save_format = QComboBox()
        self.save_format.addItem("OME-Zarr", "omezarr")
        self.save_format.addItem("TIFF", "tiff")
        save_layout.addRow(QLabel("Save Format:"), self.save_format)
        
        save_group.setLayout(save_layout)
        self.fast_acq_layout.addWidget(save_group)
        
        # Frame Generator Section
        generator_group = QGroupBox("Frame Generator")
        generator_layout = QFormLayout()
        
        self.num_frames = QSpinBox()
        self.num_frames.setMaximum(100000)
        self.num_frames.setValue(1000)
        generator_layout.addRow(QLabel("Number of Frames:"), self.num_frames)
        
        self.timeout = QDoubleSpinBox()
        self.timeout.setMaximum(60.0)
        self.timeout.setValue(10.0)
        generator_layout.addRow(QLabel("Timeout (s):"), self.timeout)
        
        generator_group.setLayout(generator_layout)
        self.fast_acq_layout.addWidget(generator_group)
        
        # Control Buttons Section
        controls_group = QGroupBox("Acquisition Controls")
        controls_layout = QVBoxLayout()
        
        # Main acquisition buttons
        acq_buttons_layout = QHBoxLayout()
        self.start_fast_acq_btn = QPushButton("Start Fast Acquisition")
        self.start_fast_acq_btn.clicked.connect(self._startFastAcquisition)
        self.stop_fast_acq_btn = QPushButton("Stop Fast Acquisition")
        self.stop_fast_acq_btn.clicked.connect(self._stopFastAcquisition)
        self.stop_fast_acq_btn.setEnabled(False)
        
        acq_buttons_layout.addWidget(self.start_fast_acq_btn)
        acq_buttons_layout.addWidget(self.stop_fast_acq_btn)
        controls_layout.addLayout(acq_buttons_layout)
        
        # Frame generator and status buttons
        util_buttons_layout = QHBoxLayout()
        self.get_frames_btn = QPushButton("Get Frame Generator")
        self.get_frames_btn.clicked.connect(self._getFrameGenerator)
        self.get_status_btn = QPushButton("Get Status")
        self.get_status_btn.clicked.connect(self._getStatus)
        self.trigger_recon_btn = QPushButton("Trigger Reconstruction")
        self.trigger_recon_btn.clicked.connect(self._triggerReconstruction)
        
        util_buttons_layout.addWidget(self.get_frames_btn)
        util_buttons_layout.addWidget(self.get_status_btn)
        util_buttons_layout.addWidget(self.trigger_recon_btn)
        controls_layout.addLayout(util_buttons_layout)
        
        controls_group.setLayout(controls_layout)
        self.fast_acq_layout.addWidget(controls_group)
        
        # Status Display Section
        status_group = QGroupBox("Status Information")
        status_layout = QVBoxLayout()
        
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(100)
        self.status_display.setReadOnly(True)
        self.status_display.setPlainText("Ready for fast acquisition...")
        
        status_layout.addWidget(self.status_display)
        status_group.setLayout(status_layout)
        self.fast_acq_layout.addWidget(status_group)
        
        # Add stretch to push everything to top
        self.fast_acq_layout.addStretch()

    def _startFastAcquisition(self):
        """Start fast STORM acquisition with current parameters"""
        params = {
            'session_id': self.session_id_input.text() or None,
            'crop_x': self.crop_x.value() if self.crop_enabled.isChecked() else None,
            'crop_y': self.crop_y.value() if self.crop_enabled.isChecked() else None,
            'crop_width': self.crop_width.value() if self.crop_enabled.isChecked() else None,
            'crop_height': self.crop_height.value() if self.crop_enabled.isChecked() else None,
            'save_path': self.save_path_input.text() if self.save_enabled.isChecked() else None,
            'save_format': self.save_format.currentData()
        }
        
        self.start_fast_acq_btn.setEnabled(False)
        self.stop_fast_acq_btn.setEnabled(True)
        self.status_display.setPlainText("Starting fast acquisition...")
        
        self.sigStartFastAcquisition.emit(params)

    def _stopFastAcquisition(self):
        """Stop fast STORM acquisition"""
        self.start_fast_acq_btn.setEnabled(True)
        self.stop_fast_acq_btn.setEnabled(False)
        self.status_display.setPlainText("Stopping fast acquisition...")
        
        self.sigStopFastAcquisition.emit()

    def _getFrameGenerator(self):
        """Request frame generator with current parameters"""
        num_frames = self.num_frames.value()
        timeout = self.timeout.value()
        
        self.status_display.setPlainText(f"Requesting {num_frames} frames with {timeout}s timeout...")
        
        self.sigGetFrameGenerator.emit(num_frames, timeout)

    def _getStatus(self):
        """Request current acquisition status"""
        self.status_display.setPlainText("Requesting status...")
        self.sigGetStatus.emit()

    def _triggerReconstruction(self):
        """Trigger single frame reconstruction"""
        self.status_display.setPlainText("Triggering reconstruction...")
        self.sigTriggerReconstruction.emit()

    def updateStatus(self, status_text: str):
        """Update the status display"""
        self.status_display.setPlainText(status_text)

    def setAcquisitionState(self, active: bool):
        """Update UI based on acquisition state"""
        self.start_fast_acq_btn.setEnabled(not active)
        self.stop_fast_acq_btn.setEnabled(active)


    def getImage(self):
        if self.layer is not None:
            return self.img.image

    def setImage(self, im):
        if self.layer is None or self.layer.name not in self.viewer.layers:
            self.layer = self.viewer.add_image(im, rgb=False, name="STORMRecon", blending='additive')
        self.layer.data = im

    def localize(self):
        if self.loc_btn.text() == "Localize":
            self.sigShowToggled.emit(True)
            self.loc_btn.setText("Stop Localizing")
        else:
            self.sigShowToggled.emit(False)
            self.loc_btn.setText("Localize")
