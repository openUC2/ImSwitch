"""useq / pymmcore-plus MDA sequences.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
from datetime import datetime
import time
from fastapi import HTTPException
import numpy as np
from typing import Dict, Any

import os
import threading

from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowContext
from imswitch.imcommon.model import dirtools, APIExport

from imswitch.imcontrol.controller.controllers.experiment_controller import MDASequenceInfo, MDASequenceRequest


class MDAMixin:
    """useq / pymmcore-plus MDA sequences."""

    @APIExport()
    def get_mda_capabilities(self) -> Dict[str, Any]:
        """Get information about MDA capabilities and available channels."""
        return {
            "mda_available": self.mda_manager.is_available(),
            "available_channels": self.allIlluNames,
            "available_detectors": self._master.detectorsManager.getAllDeviceNames(),
            "stage_available": self.mStage is not None
        }

    @APIExport(requestType="POST")
    def start_mda_experiment(self, request: MDASequenceRequest) -> Dict[str, Any]:
        """
        Start an MDA experiment using useq-schema.

        This provides a modern, standardized interface for multi-dimensional
        acquisition experiments.
        """
        if not self.mda_manager.is_available():
            raise HTTPException(status_code=400, detail="useq-schema not available")

        # Check if another workflow is running
        if self.workflow_manager.get_status()["status"] in ["running", "paused"]:
            raise HTTPException(status_code=400, detail="Another workflow is already running.")

        try:
            # Convert channel configurations to useq format
            channel_names = [ch.name for ch in request.channels]
            exposure_times = {ch.name: ch.exposure for ch in request.channels}

            # Create MDA sequence
            sequence = self.mda_manager.create_simple_sequence(
                channels=channel_names,
                z_range=request.z_range,
                z_step=request.z_step,
                time_points=request.time_points,
                time_interval=request.time_interval,
                exposure_times=exposure_times
            )

            # Get sequence info for logging/validation
            seq_info = self.mda_manager.get_sequence_info(sequence)
            self._logger.info(f"Starting MDA experiment: {seq_info}")

            # Start the detector if not already running
            if not self.mDetector._running:
                self.mDetector.startAcquisition()

            # Create controller function mapping for MDA conversion
            controller_functions = {
                'move_stage_xy': self.move_stage_xy,
                'move_stage_z': self.move_stage_z,
                'set_laser_power': self.set_laser_power,
                'set_detector_parameter': self.set_detector_parameter,
                'snap_image': self.snap_image_with_metadata
            }

            # Convert MDA sequence to workflow steps
            workflow_steps = self.mda_manager.convert_sequence_to_workflow_steps(
                sequence, controller_functions
            )

            # Setup data directory
            if request.save_directory:
                dirPath = request.save_directory
            else:
                timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                drivePath = dirtools.UserFileDirs.Data
                dirPath = os.path.join(drivePath, 'MDAExperiments', timeStamp)

            os.makedirs(dirPath, exist_ok=True)

            # Create workflow progress handler
            def sendProgress(payload):
                self.sigExperimentWorkflowUpdate.emit()

            # Create workflow and context
            from imswitch.imcontrol.model.managers.WorkflowManager import Workflow, WorkflowContext
            wf = Workflow(workflow_steps, self.workflow_manager)
            context = WorkflowContext()

            # Set metadata
            context.set_metadata("experiment_name", request.experiment_name)
            context.set_metadata("sequence_info", seq_info)
            context.set_metadata("save_directory", dirPath)
            context.set_metadata("channels", channel_names)
            context.set_metadata("experiment_start_time", time.time())

            context.on("progress", sendProgress)

            # Start the workflow
            result = self.workflow_manager.start_workflow(wf, context)

            return {
                "status": result["status"],
                "sequence_info": seq_info,
                "save_directory": dirPath,
                "estimated_duration_minutes": seq_info["estimated_duration_minutes"]
            }

        except Exception as e:
            self._logger.error(f"Error starting MDA experiment: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error starting MDA experiment: {str(e)}")

    @APIExport(requestType="POST")
    def get_mda_sequence_info(self, request: MDASequenceRequest) -> MDASequenceInfo:
        """Get information about an MDA sequence without starting it."""
        if not self.mda_manager.is_available():
            raise HTTPException(status_code=400, detail="useq-schema not available")

        try:
            # Convert channel configurations to useq format
            channel_names = [ch.name for ch in request.channels]
            exposure_times = {ch.name: ch.exposure for ch in request.channels}

            # Create MDA sequence
            sequence = self.mda_manager.create_simple_sequence(
                channels=channel_names,
                z_range=request.z_range,
                z_step=request.z_step,
                time_points=request.time_points,
                time_interval=request.time_interval,
                exposure_times=exposure_times
            )

            # Get sequence info
            info = self.mda_manager.get_sequence_info(sequence)
            return MDASequenceInfo(**info)

        except Exception as e:
            self._logger.error(f"Error getting MDA sequence info: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting MDA sequence info: {str(e)}")

    @APIExport(requestType="POST")
    def run_native_mda_sequence(self, sequence_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a native useq-schema MDASequence from JSON.

        This endpoint accepts a native useq.MDASequence serialized as JSON (dict),
        following the EXACT pattern from pymmcore-plus and raman-mda-engine.

        Args:
            sequence_dict: Native useq.MDASequence serialized to dict/JSON with fields:
                - metadata: Dict with arbitrary experiment metadata
                - stage_positions: List of (x, y, z) tuples or AbsolutePosition dicts
                - grid_plan: Dict with GridRowsColumns configuration
                - channels: List of Channel dicts with 'config' and optional 'exposure'
                - time_plan: Dict with 'interval' and 'loops' for TIntervalLoops
                - z_plan: Dict with 'range' and 'step' for ZRangeAround
                - autofocus_plan: Dict with autofocus configuration
                - axis_order: String like "tpcz" defining acquisition order
                - keep_shutter_open_across: Tuple of axes to keep shutter open

        Returns:
            Dict with execution status and sequence information

        Example request body:
            {
                "metadata": {"experiment": "test", "user": "researcher"},
                "stage_positions": [[100.0, 100.0, 30.0], [200.0, 150.0, 35.0]],
                "channels": [
                    {"config": "BF", "exposure": 50.0},
                    {"config": "DAPI", "exposure": 100.0}
                ],
                "time_plan": {"interval": 1, "loops": 20},
                "z_plan": {"range": 4.0, "step": 0.5},
                "axis_order": "tpcz"
            }
        """
        if not self.mda_manager.is_available():
            raise HTTPException(status_code=400, detail="useq-schema not available. Install with: pip install useq-schema")

        try:
            from useq import MDASequence

            # Parse the native useq-schema JSON into an MDASequence object
            # useq-schema's pydantic models can parse from dict
            self._logger.info(f"Received native MDA sequence: {sequence_dict.keys()}")
            sequence = MDASequence(**sequence_dict)

            self._logger.info(f"Parsed MDASequence: {len(list(sequence))} events, axis_order={sequence.axis_order}")

            # Register the MDA manager with ImSwitch hardware
            if not self.mda_manager._detector_manager:
                self.mda_manager.register(
                    detector_manager=self._master.detectorsManager,
                    positioners_manager=self._master.positionersManager,
                    lasers_manager=self._master.lasersManager,
                    autofocus_manager=getattr(self._master, 'autofocusManager', None)
                )
                self._logger.info("Registered MDA engine with ImSwitch hardware managers")

            # Get sequence info
            seq_info = self.mda_manager.get_sequence_info(sequence)
            self._logger.info(f"Sequence info: {seq_info}")

            # Setup data directory
            metadata = sequence.metadata if hasattr(sequence, 'metadata') and sequence.metadata else {}
            experiment_name = metadata.get('experiment_name', 'MDA_Experiment')

            timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drivePath = dirtools.UserFileDirs.Data
            dirPath = os.path.join(drivePath, 'NativeMDA', experiment_name, timeStamp)

            created_new = not os.path.exists(dirPath)
            os.makedirs(dirPath, exist_ok=True)
            if created_new:
                self._logger.info(f"Created output directory: {dirPath}")

            # Run the MDA sequence using the native engine
            # This runs in a background thread to not block the API
            def run_sequence():
                try:
                    self._logger.info("Starting native MDA sequence execution")
                    self.mda_manager.run_mda(sequence, output_path=dirPath)
                    self._logger.info("Native MDA sequence completed successfully")
                except Exception as e:
                    self._logger.error(f"Error during MDA execution: {str(e)}", exc_info=True)

            # Start execution thread
            thread = threading.Thread(target=run_sequence, daemon=False)
            thread.start()

            return {
                "status": "started",
                "sequence_info": seq_info,
                "save_directory": dirPath,
                "estimated_duration_minutes": seq_info["estimated_duration_minutes"],
                "message": "Native MDA sequence started in background thread"
            }

        except Exception as e:
            self._logger.error(f"Error starting native MDA sequence: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error starting native MDA sequence: {str(e)}")

    def snap_image_with_metadata(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Snap an image with MDA metadata."""
        # This is a wrapper around the existing snap functionality
        # that includes MDA-specific metadata handling
        image = self.mDetector.getLatestFrame()

        # Store metadata with the image (implementation depends on storage backend)
        # For now, just log it
        self._logger.debug(f"Captured MDA image with metadata: {metadata}")

        # Emit signal for live view update
        self.sigExperimentImageUpdate.emit(
            self.mDetector.name,
            image,
            False,  # not init
            [1, 1, 1, 1],  # scale
            True  # is current detector
        )

        return image

    # ================================================================
    # Focus Map – internal helpers
    # ================================================================
