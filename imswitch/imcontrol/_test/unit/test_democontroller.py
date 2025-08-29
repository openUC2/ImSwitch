import unittest
import os
import tempfile
import importlib.util
import sys
import pathlib
import time


# Dynamically build the path to DemoController.py based on this test file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_path = os.path.join(
    current_dir,
    "..", "..", "controller", "controllers", "DemoController.py"
)
controller_path = os.path.abspath(controller_path)

spec = importlib.util.spec_from_file_location(
    'DemoController',
    controller_path
)
demo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo_module)

DemoController = demo_module.DemoController
DemoParams = demo_module.DemoParams


class TestDemoController(unittest.TestCase):
    """Test cases for DemoController."""

    def setUp(self):
        """Set up test fixtures."""
        # Create controller instance (will run in testing mode without ImSwitch)
        self.controller = DemoController()

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop any running demo
        if self.controller.isRunning:
            self.controller.stopDemo()

    def test_controller_initialization(self):
        """Test that controller initializes correctly."""
        self.assertIsNotNone(self.controller)
        self.assertFalse(self.controller.isRunning)
        self.assertEqual(self.controller.params.scanningScheme, "random")
        self.assertEqual(self.controller.params.illuminationMode, "random")

    def test_get_demo_params(self):
        """Test getting demo parameters."""
        params = self.controller.getDemoParams()
        self.assertIsInstance(params, DemoParams)
        self.assertEqual(params.scanningScheme, "random")
        self.assertEqual(params.illuminationMode, "random")
        self.assertEqual(params.numRandomPositions, 10)

    def test_set_demo_params(self):
        """Test setting demo parameters."""
        new_params = DemoParams(
            minPosX=100.0,
            maxPosX=5000.0,
            minPosY=200.0,
            maxPosY=4000.0,
            scanningScheme="grid",
            illuminationMode="continuous",
            gridRows=5,
            gridColumns=4,
            dwellTime=1.0
        )

        result = self.controller.setDemoParams(new_params)
        self.assertTrue(result)

        # Verify parameters were set
        current_params = self.controller.getDemoParams()
        self.assertEqual(current_params.maxPosX, 5000.0)
        self.assertEqual(current_params.scanningScheme, "grid")
        self.assertEqual(current_params.illuminationMode, "continuous")
        self.assertEqual(current_params.gridRows, 5)
        self.assertEqual(current_params.gridColumns, 4)

    def test_generate_random_positions(self):
        """Test random position generation."""
        self.controller.params.scanningScheme = "random"
        self.controller.params.numRandomPositions = 5
        self.controller.params.minPosX = 0.0
        self.controller.params.maxPosX = 1000.0
        self.controller.params.minPosY = 0.0
        self.controller.params.maxPosY = 1000.0

        positions = self.controller._generatePositions()

        # Check that correct number of positions were generated
        self.assertEqual(len(positions), 5)

        # Check that positions are within range
        for pos in positions:
            self.assertTrue(0.0 <= pos[0] <= 1000.0)
            self.assertTrue(0.0 <= pos[1] <= 1000.0)

    def test_generate_grid_positions(self):
        """Test grid position generation."""
        self.controller.params.scanningScheme = "grid"
        self.controller.params.gridRows = 3
        self.controller.params.gridColumns = 3
        self.controller.params.minPosX = 0.0
        self.controller.params.maxPosX = 1000.0
        self.controller.params.minPosY = 0.0
        self.controller.params.maxPosY = 1000.0

        positions = self.controller._generatePositions()

        # Check that positions were generated (exact number depends on implementation)
        self.assertGreater(len(positions), 0)

        # Check that positions are within range
        for pos in positions:
            self.assertTrue(0.0 <= pos[0] <= 1000.0)
            self.assertTrue(0.0 <= pos[1] <= 1000.0)

    def test_generate_spiral_positions(self):
        """Test spiral position generation."""
        self.controller.params.scanningScheme = "spiral"
        self.controller.params.spiralShells = 2
        self.controller.params.minPosX = 0.0
        self.controller.params.maxPosX = 1000.0
        self.controller.params.minPosY = 0.0
        self.controller.params.maxPosY = 1000.0

        positions = self.controller._generatePositions()

        # Check that positions were generated
        self.assertGreater(len(positions), 0)

        # Check that positions are within range (with some tolerance for spiral)
        for pos in positions:
            self.assertTrue(-100.0 <= pos[0] <= 1100.0)  # Allow some margin for spiral
            self.assertTrue(-100.0 <= pos[1] <= 1100.0)

    def test_move_to_position(self):
        """Test moving to position in testing mode."""
        position = [100.0, 200.0]
        
        # Should not raise error in testing mode
        self.controller._moveToPosition(position)

    def test_illumination_control(self):
        """Test illumination control functions in testing mode."""
        # Should not raise errors in testing mode
        self.controller._setRandomIllumination()
        self.controller._setContinuousIllumination(True)
        self.controller._setContinuousIllumination(False)
        self.controller._turnOffIllumination()

    def test_start_stop_demo(self):
        """Test starting and stopping demo."""
        # Set minimal parameters for quick test
        self.controller.params.numRandomPositions = 2
        self.controller.params.dwellTime = 0.1
        self.controller.params.totalRunTime = 1.0

        # Start demo
        result = self.controller.startDemo()
        self.assertTrue(result)
        self.assertTrue(self.controller.isRunning)

        # Let it run briefly
        time.sleep(0.2)

        # Stop demo
        result = self.controller.stopDemo()
        self.assertTrue(result)
        
        # Wait a moment for cleanup
        time.sleep(0.1)
        self.assertFalse(self.controller.isRunning)

    def test_start_demo_already_running(self):
        """Test starting demo when already running."""
        # Set minimal parameters
        self.controller.params.dwellTime = 0.1
        self.controller.params.totalRunTime = 1.0

        # Start demo
        self.controller.startDemo()
        self.assertTrue(self.controller.isRunning)

        # Try to start again
        result = self.controller.startDemo()
        self.assertFalse(result)  # Should fail

        # Clean up
        self.controller.stopDemo()

    def test_stop_demo_not_running(self):
        """Test stopping demo when not running."""
        self.assertFalse(self.controller.isRunning)
        
        result = self.controller.stopDemo()
        self.assertFalse(result)  # Should fail

    def test_demo_results_structure(self):
        """Test that results have correct structure."""
        results = self.controller.getDemoResults()

        # Check that results object has expected attributes
        self.assertIsInstance(results.totalPositions, int)
        self.assertIsInstance(results.currentPosition, int)
        self.assertIsInstance(results.currentCoordinates, list)
        self.assertIsInstance(results.isRunning, bool)
        self.assertIsInstance(results.elapsedTime, float)
        self.assertIsInstance(results.startTime, str)
        self.assertIsInstance(results.currentIllumination, dict)

    def test_demo_params_validation(self):
        """Test demo parameter validation."""
        # Test valid parameters
        valid_params = DemoParams(
            minPosX=0.0,
            maxPosX=1000.0,
            scanningScheme="grid",
            illuminationMode="continuous"
        )
        self.assertIsInstance(valid_params, DemoParams)

        # Test parameter setting and getting
        result = self.controller.setDemoParams(valid_params)
        self.assertTrue(result)
        
        retrieved_params = self.controller.getDemoParams()
        self.assertEqual(retrieved_params.scanningScheme, "grid")
        self.assertEqual(retrieved_params.illuminationMode, "continuous")

    def test_short_demo_execution(self):
        """Test a complete short demo execution."""
        # Set parameters for very short demo
        self.controller.params.scanningScheme = "random"
        self.controller.params.numRandomPositions = 3
        self.controller.params.dwellTime = 0.05  # Very short dwell time
        self.controller.params.totalRunTime = 0.5  # Very short total time
        self.controller.params.minPosX = 0.0
        self.controller.params.maxPosX = 100.0
        self.controller.params.minPosY = 0.0
        self.controller.params.maxPosY = 100.0

        # Start demo
        result = self.controller.startDemo()
        self.assertTrue(result)
        
        # Wait for completion
        start_time = time.time()
        while self.controller.isRunning and (time.time() - start_time) < 2.0:
            time.sleep(0.01)
        
        # Check that demo completed
        self.assertFalse(self.controller.isRunning)
        
        # Check results
        results = self.controller.getDemoResults()
        self.assertGreater(results.elapsedTime, 0.0)


if __name__ == '__main__':
    unittest.main()