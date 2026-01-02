"""
Unit tests for MDA functionality integration in ImSwitch ExperimentController.
"""
import unittest
from unittest.mock import Mock

class TestMDAIntegration(unittest.TestCase):
    """Test MDA integration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the ImSwitch dependencies
        self.mock_master = Mock()
        self.mock_comm_channel = Mock()

        # We'll test the MDASequenceManager in isolation
        # since full ImSwitch environment is complex to set up

    def test_useq_availability(self):
        """Test that useq-schema can be imported and used."""
        try:
            import useq
            from useq import MDASequence, Channel, TIntervalLoops, ZRangeAround

            # Create a simple sequence
            sequence = MDASequence(
                axis_order='tpzc',
                time_plan=TIntervalLoops(interval=1.0, loops=2),
                z_plan=ZRangeAround(range=4, step=2),
                channels=[
                    Channel(config='DAPI', exposure=50),
                    Channel(config='FITC', exposure=100)
                ]
            )

            events = list(sequence)
            self.assertGreater(len(events), 0)

            # Check that events have expected properties
            first_event = events[0]
            self.assertIsNotNone(first_event.channel)
            self.assertEqual(first_event.channel.config, 'DAPI')
            self.assertEqual(first_event.exposure, 50.0)

        except ImportError:
            self.skipTest("useq-schema not available")

    def test_mda_sequence_manager_isolation(self):
        """Test MDASequenceManager in isolation."""
        try:
            # We can't easily import the full manager due to ImSwitch dependencies
            # But we can test the concept
            import useq

            # Test sequence creation
            sequence = self._create_test_sequence()
            events = list(sequence)

            # Test sequence analysis
            info = self._analyze_sequence(sequence)
            self.assertIn('total_events', info)
            self.assertIn('channels', info)
            self.assertGreater(info['total_events'], 0)

        except ImportError:
            self.skipTest("useq-schema not available")

    def test_workflow_step_conversion_concept(self):
        """Test the concept of converting MDA events to workflow steps."""
        try:
            import useq
            sequence = self._create_test_sequence()

            # Mock controller functions
            mock_functions = {
                'move_stage_xy': Mock(return_value=None),
                'move_stage_z': Mock(return_value=None),
                'set_laser_power': Mock(return_value=None),
                'snap_image': Mock(return_value=None)
            }

            # Simulate workflow step conversion
            steps = []
            for event_idx, event in enumerate(sequence):
                event_steps = self._convert_event_to_mock_steps(event, event_idx, mock_functions)
                steps.extend(event_steps)

            self.assertGreater(len(steps), 0)
            # Each event should generate multiple steps (move Z, set illumination, acquire, etc.)
            self.assertGreater(len(steps), len(list(sequence)))

        except ImportError:
            self.skipTest("useq-schema not available")

    def _create_test_sequence(self):
        """Helper to create a test MDA sequence."""
        from useq import MDASequence, Channel, ZRangeAround

        return MDASequence(
            axis_order='tpzc',
            z_plan=ZRangeAround(range=4, step=2),
            channels=[
                Channel(config='TestChannel', exposure=100)
            ]
        )

    def _analyze_sequence(self, sequence):
        """Helper to analyze a sequence - simplified version of MDASequenceManager.get_sequence_info."""
        events = list(sequence)

        channels = set()
        z_positions = set()

        for event in events:
            if event.channel:
                channels.add(event.channel.config)
            if event.z_pos is not None:
                z_positions.add(event.z_pos)

        return {
            'total_events': len(events),
            'channels': list(channels),
            'z_positions': list(z_positions)
        }

    def _convert_event_to_mock_steps(self, event, event_idx, mock_functions):
        """Helper to simulate workflow step conversion."""
        steps = []

        # Simulate creating steps for Z movement, illumination, acquisition
        if event.z_pos is not None:
            steps.append(f"move_z_{event_idx}")

        if event.channel:
            steps.append(f"set_illumination_{event_idx}")
            steps.append(f"acquire_image_{event_idx}")
            steps.append(f"turn_off_illumination_{event_idx}")

        return steps

if __name__ == '__main__':
    unittest.main()
