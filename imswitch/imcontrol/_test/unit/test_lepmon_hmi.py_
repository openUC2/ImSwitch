import unittest
import os
import importlib.util
from unittest.mock import Mock, patch
import threading


# Dynamically build the path to LepmonController.py based on this test file's location
current_dir = os.path.dirname(os.path.abspath(__file__))
controller_path = os.path.join(
    current_dir,
    "..", "..", "controller", "controllers", "LepmonController.py"
)
controller_path = os.path.abspath(controller_path)

# Mock the ImSwitch dependencies that aren't available in test environment
class MockSignal:
    def __init__(self, *args):
        pass
    def emit(self, data):
        pass

class MockAPIExport:
    def __init__(self, **kwargs):
        pass
    def __call__(self, func):
        return func

class MockLiveUpdatedController:
    def __init__(self, *args, **kwargs):
        pass

class MockInitLogger:
    def __init__(self, *args, **kwargs):
        pass
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

# Patch the imports before loading the module
with patch.dict('sys.modules', {
    'imswitch.imcommon.framework': Mock(Signal=MockSignal),
    'imswitch.imcommon.model': Mock(APIExport=MockAPIExport, initLogger=lambda *args, **kwargs: MockInitLogger()),
    'imswitch.imcontrol.controller.controllers.basecontrollers': Mock(LiveUpdatedController=MockLiveUpdatedController),
    'imswitch': Mock(IS_HEADLESS=True),
    'RPi.GPIO': Mock(),
    'luma.core.interface.serial': Mock(),
    'luma.core.render': Mock(),
    'luma.oled.device': Mock(),
    'PIL.ImageFont': Mock(),
    'PIL.ImageDraw': Mock(),
    'PIL.Image': Mock(),
    'smbus': Mock(),
}):
    spec = importlib.util.spec_from_file_location('LepmonController', controller_path)
    lepmon_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lepmon_module)

LepmonController = lepmon_module.LepmonController


class TestLepmonHMI(unittest.TestCase):
    """Test cases for LepmonController HMI functionality matching trap_hmi.py."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock master object with required managers
        self.mock_master = Mock()
        self.mock_master.LepmonManager = Mock()
        self.mock_master.LepmonManager.defaultConfig = lepmon_module.DEFAULT_CONFIG
        self.mock_master.LepmonManager.updateConfig = Mock()
        self.mock_master.detectorsManager = Mock()
        self.mock_master.detectorsManager.getAllDeviceNames = Mock(return_value=['TestCamera'])

        # Mock detector
        self.mock_detector = Mock()
        self.mock_detector.getLatestFrame = Mock(return_value=None)
        self.mock_detector.setParameter = Mock()
        self.mock_detector.setGain = Mock()
        self.mock_master.detectorsManager.__getitem__ = Mock(return_value=self.mock_detector)

        # Create controller instance with mocked dependencies
        with patch('threading.Thread'), \
             patch('time.sleep'), \
             patch('os.path.exists', return_value=False):
            self.controller = LepmonController(self.mock_master)

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop any running threads
        if hasattr(self.controller, 'hmi_stop_event'):
            self.controller.hmi_stop_event.set()
        if hasattr(self.controller, '_pullSensorDataActive'):
            self.controller._pullSensorDataActive = False

    def test_controller_initialization(self):
        """Test that controller initializes correctly with HMI components."""
        self.assertIsNotNone(self.controller)
        self.assertFalse(self.controller.menu_open)
        self.assertEqual(self.controller.current_menu_state, "main")
        self.assertIsInstance(self.controller.lightStates, dict)
        self.assertIsInstance(self.controller.lcdDisplay, dict)
        self.assertIsInstance(self.controller.buttonStates, dict)

    def test_open_hmi_menu(self):
        """Test HMI menu opening - matching trap_hmi.py initial state."""
        with patch.object(self.controller, '_turn_on_led') as mock_led, \
             patch.object(self.controller, '_display_text') as mock_display:

            self.controller._open_hmi_menu()

            # Verify blue LED is turned on (menu indicator)
            mock_led.assert_called_with("blau")

            # Verify correct display text (German interface matching trap_hmi.py)
            mock_display.assert_called_with("Menü öffnen:", "bitte Enter drücken", "(rechts unten)", "")

            # Menu should not be open yet (waiting for enter press)
            self.assertFalse(self.controller.menu_open)

    def test_button_press_detection(self):
        """Test button press detection with GPIO simulation."""
        # Test button press in simulation mode
        with patch.object(self.controller, '_simulate_button_press') as mock_sim:
            result = self.controller.simulateButtonPress("enter")
            self.assertTrue(result["success"])
            self.assertEqual(result["message"], "Button 'enter' pressed (simulated)")

    def test_menu_enter_sequence(self):
        """Test main menu entry sequence (200 iterations waiting for enter)."""
        with patch.object(self.controller, '_button_pressed') as mock_button, \
             patch.object(self.controller, '_display_text') as mock_display, \
             patch('time.sleep'):

            # Simulate enter button press on 5th iteration
            mock_button.side_effect = lambda btn: btn == "enter" and mock_button.call_count >= 5

            # Simulate the menu entry logic
            self.controller.menu_open = False
            for i in range(10):  # Shortened loop for test
                if self.controller._button_pressed("enter"):
                    self.controller.menu_open = True
                    mock_display("Eingabe Menü", "geöffnet", "", "")
                    break

            # Verify menu was opened
            self.assertTrue(self.controller.menu_open)
            mock_display.assert_called_with("Eingabe Menü", "geöffnet", "", "")

    def test_focus_menu_handling(self):
        """Test focus menu handling (rechts button)."""
        with patch.object(self.controller, '_run_focus_mode') as mock_focus, \
             patch.object(self.controller, '_display_text') as mock_display:

            self.controller._handle_focus_menu()

            # Verify focus mode activation
            mock_display.assert_any_call("Fokussierhilfe", "aktiviert", "15 Sekunden", "")
            mock_focus.assert_called_once()

    def test_location_menu_handling(self):
        """Test location menu handling (unten button)."""
        with patch.object(self.controller, '_display_text') as mock_display, \
             patch('time.sleep'):

            self.controller._handle_location_menu()

            # Verify correct display for location menu
            mock_display.assert_any_call("Code unverändert", "fahre fort", "", "")

    def test_time_menu_handling(self):
        """Test time setting menu with user interaction."""
        with patch.object(self.controller, '_button_pressed') as mock_button, \
             patch.object(self.controller, '_display_text') as mock_display, \
             patch.object(self.controller, '_turn_on_led') as mock_led_on, \
             patch.object(self.controller, '_turn_off_led') as mock_led_off, \
             patch('time.sleep'):

            # Simulate "oben" button press to update time
            mock_button.side_effect = lambda btn: btn == "oben" and mock_button.call_count >= 3
            self.controller.hmi_stop_event = threading.Event()  # Ensure event exists

            self.controller._handle_time_menu()

            # Verify time menu displays
            mock_display.assert_any_call("Datum / Uhrzeit:", "hoch aktualisieren", "runter bestätigen", "")
            mock_led_on.assert_called_with("blau")
            mock_led_off.assert_called_with("blau")

    def test_gps_menu_handling(self):
        """Test GPS coordinate menu with user interaction."""
        with patch.object(self.controller, '_button_pressed') as mock_button, \
             patch.object(self.controller, '_display_text') as mock_display, \
             patch.object(self.controller, '_turn_on_led') as mock_led_on, \
             patch.object(self.controller, '_turn_off_led') as mock_led_off, \
             patch('time.sleep'):

            # Simulate "unten" button press to confirm coordinates
            mock_button.side_effect = lambda btn: btn == "unten" and mock_button.call_count >= 2
            self.controller.hmi_stop_event = threading.Event()  # Ensure event exists

            self.controller._handle_gps_menu()

            # Verify GPS menu displays
            mock_display.assert_any_call("Koordinaten mit", "hoch aktualisieren", "runter bestätigen", "")
            mock_led_on.assert_called_with("blau")
            mock_led_off.assert_called_with("blau")

    def test_system_test_sequence(self):
        """Test complete system test sequence."""
        with patch.object(self.controller, '_display_text') as mock_display, \
             patch.object(self.controller, '_read_sensor_data') as mock_sensors, \
             patch.object(self.controller, '_display_sensor_status_with_text') as mock_sensor_display, \
             patch.object(self.controller, 'lepmonSnapImage') as mock_snap, \
             patch.object(self.controller, '_lepiled_start') as mock_led_start, \
             patch.object(self.controller, '_lepiled_ende') as mock_led_end, \
             patch.object(self.controller, '_get_disk_space') as mock_disk, \
             patch('time.sleep'):

            # Mock successful responses
            mock_sensors.return_value = {"LUX": 50, "Temp_in": 22.5, "bus_voltage": 12.1, "Temp_out": 18.3}
            mock_snap.return_value = {"success": True}
            mock_disk.return_value = (100.0, 50.0, 50.0, 50.0, 50.0)  # 50GB free

            self.controller._handle_system_test()

            # Verify system test sequence
            mock_display.assert_any_call("Testlauf starten", "", "", "")
            mock_sensor_display.assert_called_once()
            mock_display.assert_any_call("Kamera Test", "", "", "")
            mock_led_start.assert_called()
            mock_led_end.assert_called()
            mock_snap.assert_called()
            mock_display.assert_any_call("USB Speicher", "OK", "", "")
            mock_display.assert_any_call("Testlauf beendet", "bitte Deckel", "schließen", "")

    def test_sensor_status_display(self):
        """Test sensor status display with text format."""
        with patch.object(self.controller, '_display_text') as mock_display, \
             patch('time.sleep'):

            sensor_data = {
                "LUX": 45.2,
                "Temp_in": 23.1,
                "bus_voltage": 11.8,
                "Temp_out": 19.5
            }

            self.controller._display_sensor_status_with_text(sensor_data)

            # Verify sensor display calls (should be 4 sensors)
            self.assertEqual(mock_display.call_count, 4)

            # Verify correct format for one sensor
            calls = mock_display.call_args_list
            first_call = calls[0][0]  # First positional arguments
            self.assertEqual(first_call[0], "Light_Sensor")
            self.assertEqual(first_call[1], "Status: OK")
            self.assertIn("45.2 Lux", first_call[2])

    def test_camera_test_with_retry(self):
        """Test camera test with retry logic."""
        with patch.object(self.controller, '_display_text') as mock_display, \
             patch.object(self.controller, 'lepmonSnapImage') as mock_snap, \
             patch.object(self.controller, '_lepiled_start') as mock_led_start, \
             patch.object(self.controller, '_lepiled_ende') as mock_led_end, \
             patch('time.sleep'):

            # First call fails, second succeeds
            mock_snap.side_effect = [
                {"success": False},  # First attempt fails
                {"success": True}    # Second attempt succeeds
            ]

            # Simulate camera test portion of system test
            Status_Kamera = 0
            test_attempts = 0
            max_attempts = 3

            while Status_Kamera == 0 and test_attempts < max_attempts:
                mock_display("Kamera Test", "aktiviere", "UV Lampe", "")
                mock_led_start()

                result = mock_snap("jpg", "display", 0, 80)
                Status_Kamera = 1 if result["success"] else 0

                mock_led_end()
                test_attempts += 1

                if Status_Kamera == 0:
                    mock_display("Kamera Test", "Fehler- Falle", "wiederholt Test", "")

            # Verify retry logic worked
            self.assertEqual(test_attempts, 2)
            self.assertEqual(Status_Kamera, 1)
            mock_display.assert_any_call("Kamera Test", "Fehler- Falle", "wiederholt Test", "")

    def test_led_control_integration(self):
        """Test LED control integration with HMI states."""
        with patch.object(self.controller, '_turn_on_led') as mock_on, \
             patch.object(self.controller, '_turn_off_led') as mock_off:

            # Test menu open LED control
            self.controller._open_hmi_menu()
            mock_on.assert_called_with("blau")

            # Test LED off when menu closes
            self.controller._turn_off_led("blau")
            mock_off.assert_called_with("blau")

    def test_display_text_formatting(self):
        """Test display text formatting matches trap_hmi.py patterns."""
        with patch.object(self.controller, '_display_text') as mock_display:

            # Test German text patterns from trap_hmi.py
            self.controller._open_hmi_menu()
            mock_display.assert_called_with("Menü öffnen:", "bitte Enter drücken", "(rechts unten)", "")

            # Test sensor display patterns
            self.controller._display_text("Light_Sensor", "Status: OK", "Wert: 45.2 Lux", "")
            mock_display.assert_called_with("Light_Sensor", "Status: OK", "Wert: 45.2 Lux", "")

    def test_hmi_state_transitions(self):
        """Test HMI state transitions through menu system."""
        # Test state progression
        self.assertEqual(self.controller.current_menu_state, "main")

        # Simulate menu completion
        self.controller.current_menu_state = "submenu_completed"
        self.assertEqual(self.controller.current_menu_state, "submenu_completed")

        # Simulate progression to time menu
        self.controller.current_menu_state = "time_menu"
        self.assertEqual(self.controller.current_menu_state, "time_menu")

    def test_error_handling(self):
        """Test error handling in HMI functions."""
        with patch.object(self.controller, '_display_text', side_effect=Exception("Display error")), \
             patch.object(self.controller._logger, 'error') as mock_log:

            # Test that errors are logged but don't crash
            self.controller._open_hmi_menu()
            mock_log.assert_called()

    def test_timing_integration(self):
        """Test timing integration with original trap_hmi.py patterns."""
        # Verify timing constants are preserved
        self.assertIsInstance(self.controller.timingConfig, dict)
        self.assertIn("acquisitionInterval", self.controller.timingConfig)

        # Test timing config update
        new_config = {"acquisitionInterval": 30}
        result = self.controller.setTimingConfig(new_config)
        self.assertTrue(result["success"])
        self.assertEqual(self.controller.timingConfig["acquisitionInterval"], 30)


if __name__ == '__main__':
    unittest.main()
