import unittest
import os
import tempfile
import importlib.util
import sys
import pathlib


# Dynamically build the path to StresstestController.py based on this test file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_path = os.path.join(
    current_dir,
    "..", "..", "controller", "controllers", "StresstestController.py"
)
controller_path = os.path.abspath(controller_path)

spec = importlib.util.spec_from_file_location(
    'StresstestController',
    controller_path
)
stresstest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stresstest_module)

StresstestController = stresstest_module.StresstestController
StresstestParams = stresstest_module.StresstestParams


class TestStresstestController(unittest.TestCase):
    """Test cases for StresstestController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Create controller instance (will run in testing mode without ImSwitch)
        self.controller = StresstestController()

        # Override output path to use temp directory
        self.controller.params.outputPath = self.temp_dir

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_controller_initialization(self):
        """Test that controller initializes correctly."""
        self.assertIsNotNone(self.controller)
        self.assertFalse(self.controller.isRunning)

    def test_get_stresstest_params(self):
        """Test getting stress test parameters."""
        params = self.controller.getStresstestParams()
        self.assertIsInstance(params, StresstestParams)
        self.assertEqual(params.numRandomPositions, 10)
        self.assertEqual(params.numCycles, 5)

    def test_set_stresstest_params(self):
        """Test setting stress test parameters."""
        new_params = StresstestParams(
            minPosX=0.0,
            maxPosX=5000.0,
            numRandomPositions=5,
            numCycles=3,
            timeInterval=30.0
        )

        result = self.controller.setStresstestParams(new_params)
        self.assertTrue(result)

        # Verify parameters were set
        current_params = self.controller.getStresstestParams()
        self.assertEqual(current_params.maxPosX, 5000.0)
        self.assertEqual(current_params.numRandomPositions, 5)
        self.assertEqual(current_params.numCycles, 3)

    def test_generate_random_positions(self):
        """Test random position generation."""
        self.controller.params.numCycles = 2
        self.controller.params.numRandomPositions = 3
        self.controller.params.minPosX = 0.0
        self.controller.params.maxPosX = 1000.0
        self.controller.params.minPosY = 0.0
        self.controller.params.maxPosY = 1000.0

        self.controller._generateRandomPositions()

        # Check that correct number of positions were generated
        self.assertEqual(len(self.controller.target_positions), 2)  # 2 cycles
        self.assertEqual(len(self.controller.target_positions[0]), 3)  # 3 positions per cycle
        self.assertEqual(len(self.controller.target_positions[1]), 3)  # 3 positions per cycle

        # Check that positions are within range
        for cycle_positions in self.controller.target_positions:
            for pos in cycle_positions:
                self.assertTrue(0.0 <= pos[0] <= 1000.0)
                self.assertTrue(0.0 <= pos[1] <= 1000.0)

    def test_hardware_validation(self):
        """Test hardware validation in testing mode."""
        # Should pass in testing mode
        self.assertTrue(self.controller._validateHardware())

    def test_process_position(self):
        """Test position processing in testing mode."""
        target_pos = [100.0, 200.0]

        # Process position
        self.controller._processPosition(target_pos, cycle=0, pos_idx=0)

        # Verify results were recorded
        self.assertEqual(len(self.controller.actual_positions), 1)
        self.assertEqual(len(self.controller.position_errors), 1)
        self.assertEqual(self.controller.results.completedPositions, 1)

    def test_start_stop_stresstest(self):
        """Test starting and stopping stress test."""
        # Set minimal parameters for quick test
        self.controller.params.numCycles = 1
        self.controller.params.numRandomPositions = 2
        self.controller.params.timeInterval = 0.1
        self.controller.params.saveImages = False  # Skip image saving for test

        # Start stress test
        result = self.controller.startStresstest()
        self.assertTrue(result)
        self.assertTrue(self.controller.isRunning)

        # Stop stress test
        result = self.controller.stopStresstest()
        self.assertTrue(result)
        self.assertFalse(self.controller.isRunning)

    def test_results_structure(self):
        """Test that results have correct structure."""
        results = self.controller.getStresstestResults()

        # Check that results object has expected attributes
        self.assertIsInstance(results.totalPositions, int)
        self.assertIsInstance(results.completedPositions, int)
        self.assertIsInstance(results.averagePositionError, float)
        self.assertIsInstance(results.maxPositionError, float)
        self.assertIsInstance(results.positionErrors, list)
        self.assertIsInstance(results.targetPositions, list)
        self.assertIsInstance(results.actualPositions, list)
        self.assertIsInstance(results.timestamps, list)
        self.assertIsInstance(results.isRunning, bool)


if __name__ == '__main__':
    unittest.main()
