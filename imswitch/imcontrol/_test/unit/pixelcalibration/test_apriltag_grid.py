"""
Unit tests for AprilTag grid calibration system.

Run with: pytest test_apriltag_grid.py
"""

import pytest
import numpy as np
from imswitch.imcontrol.controller.controllers.pixelcalibration.apriltag_grid_calibrator import (
    GridConfig, AprilTagGridCalibrator
)


class TestGridConfig:
    """Test GridConfig class."""

    def test_init(self):
        """Test basic initialization."""
        config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)
        assert config.rows == 17
        assert config.cols == 25
        assert config.start_id == 0
        assert config.pitch_mm == 40.0

    def test_id_to_rowcol(self):
        """Test tag ID to (row, col) conversion."""
        config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)

        # Test first tag
        assert config.id_to_rowcol(0) == (0, 0)

        # Test first row, last column
        assert config.id_to_rowcol(24) == (0, 24)

        # Test second row, first column
        assert config.id_to_rowcol(25) == (1, 0)

        # Test middle tag
        assert config.id_to_rowcol(101) == (4, 1)  # row=4, col=1

        # Test last tag
        assert config.id_to_rowcol(424) == (16, 24)

        # Test out of range
        assert config.id_to_rowcol(425) is None
        assert config.id_to_rowcol(-1) is None

    def test_rowcol_to_id(self):
        """Test (row, col) to tag ID conversion."""
        config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)

        # Test first position
        assert config.rowcol_to_id(0, 0) == 0

        # Test first row, last column
        assert config.rowcol_to_id(0, 24) == 24

        # Test second row, first column
        assert config.rowcol_to_id(1, 0) == 25

        # Test middle position
        assert config.rowcol_to_id(4, 1) == 101

        # Test last position
        assert config.rowcol_to_id(16, 24) == 424

        # Test out of range
        assert config.rowcol_to_id(17, 0) is None
        assert config.rowcol_to_id(0, 25) is None
        assert config.rowcol_to_id(-1, 0) is None

    def test_roundtrip(self):
        """Test ID <-> (row,col) roundtrip conversion."""
        config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)

        # Test all valid IDs
        for tag_id in range(0, 425):
            rowcol = config.id_to_rowcol(tag_id)
            assert rowcol is not None
            reconstructed_id = config.rowcol_to_id(*rowcol)
            assert reconstructed_id == tag_id

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)
        data = config.to_dict()

        assert data["rows"] == 17
        assert data["cols"] == 25
        assert data["start_id"] == 0
        assert data["pitch_mm"] == 40.0

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "rows": 17,
            "cols": 25,
            "start_id": 0,
            "pitch_mm": 40.0
        }
        config = GridConfig.from_dict(data)

        assert config.rows == 17
        assert config.cols == 25
        assert config.start_id == 0
        assert config.pitch_mm == 40.0


class TestAprilTagGridCalibrator:
    """Test AprilTagGridCalibrator class."""

    @pytest.fixture
    def calibrator(self):
        """Create a calibrator instance for testing."""
        config = GridConfig(rows=17, cols=25, start_id=0, pitch_mm=40.0)
        return AprilTagGridCalibrator(config)

    def test_init(self, calibrator):
        """Test initialization."""
        assert calibrator._grid.rows == 17
        assert calibrator._grid.cols == 25
        assert calibrator._T_cam2stage is None

    def test_grid_to_stage_delta(self, calibrator):
        """Test grid displacement to stage displacement conversion."""
        # Tag 0 to tag 1 (one column right)
        delta = calibrator.grid_to_stage_delta(0, 1)
        assert delta is not None
        assert abs(delta[0] - 40000.0) < 1e-6  # 40mm = 40000 µm
        assert abs(delta[1] - 0.0) < 1e-6

        # Tag 0 to tag 25 (one row down)
        delta = calibrator.grid_to_stage_delta(0, 25)
        assert delta is not None
        assert abs(delta[0] - 0.0) < 1e-6
        assert abs(delta[1] - 40000.0) < 1e-6

        # Tag 0 to tag 101 (row=4, col=1)
        delta = calibrator.grid_to_stage_delta(0, 101)
        assert delta is not None
        assert abs(delta[0] - 40000.0) < 1e-6   # 1 column * 40mm
        assert abs(delta[1] - 160000.0) < 1e-6  # 4 rows * 40mm

        # Invalid tag IDs
        assert calibrator.grid_to_stage_delta(0, 425) is None
        assert calibrator.grid_to_stage_delta(425, 0) is None

    def test_calibrate_from_frame_insufficient_tags(self, calibrator):
        """Test calibration with insufficient tags."""
        # Only 2 tags (need at least 3)
        tags = {
            50: (100.0, 200.0),
            51: (150.0, 205.0)
        }

        result = calibrator.calibrate_from_frame(tags)
        assert "error" in result
        assert "at least 3" in result["error"].lower()

    def test_calibrate_from_frame_synthetic(self, calibrator):
        """Test calibration with synthetic perfect data."""
        # Create synthetic tags with known transformation
        # T = [[0.02, 0, 0], [0, 0.02, 0]]  (20 µm/pixel scale)
        scale_mm_per_px = 0.02

        # Generate synthetic tag positions
        tags = {}
        for tag_id in [50, 51, 52, 75, 76, 77]:
            rowcol = calibrator._grid.id_to_rowcol(tag_id)
            if rowcol is None:
                continue

            row, col = rowcol
            # Physical position in mm
            x_mm = col * calibrator._grid.pitch_mm
            y_mm = row * calibrator._grid.pitch_mm

            # Convert to pixel position using synthetic transform
            cx_px = x_mm / scale_mm_per_px
            cy_px = y_mm / scale_mm_per_px

            tags[tag_id] = (cx_px, cy_px)

        # Perform calibration
        result = calibrator.calibrate_from_frame(tags)

        assert "error" not in result
        assert result["num_tags"] == 6
        assert result["residual_um"] < 1e-6  # Should be near-zero for perfect data

        # Check transformation matrix
        T = np.array(result["T_cam2stage"])
        assert T.shape == (2, 3)

        # Check scale (should be ~0.02 mm/px = 20 µm/px)
        assert abs(T[0, 0] - scale_mm_per_px) < 1e-6
        assert abs(T[1, 1] - scale_mm_per_px) < 1e-6

        # Check rotation (should be near-zero)
        assert abs(T[0, 1]) < 1e-6
        assert abs(T[1, 0]) < 1e-6

    def test_calibrate_from_frame_with_noise(self, calibrator):
        """Test calibration with noisy synthetic data."""
        scale_mm_per_px = 0.02
        noise_px = 0.5  # 0.5 pixel noise

        np.random.seed(42)  # For reproducibility

        # Generate synthetic tags with noise
        tags = {}
        for tag_id in [50, 51, 52, 75, 76, 77, 100, 101, 102]:
            rowcol = calibrator._grid.id_to_rowcol(tag_id)
            if rowcol is None:
                continue

            row, col = rowcol
            x_mm = col * calibrator._grid.pitch_mm
            y_mm = row * calibrator._grid.pitch_mm

            cx_px = x_mm / scale_mm_per_px + np.random.randn() * noise_px
            cy_px = y_mm / scale_mm_per_px + np.random.randn() * noise_px

            tags[tag_id] = (cx_px, cy_px)

        # Perform calibration
        result = calibrator.calibrate_from_frame(tags)

        assert "error" not in result
        assert result["num_tags"] == 9

        # Residual should be on order of noise level
        # Expected residual: ~noise_px * scale = 0.5 * 20 = 10 µm
        assert result["residual_um"] < 20.0  # Allow some margin

        # Transformation should still be close to ground truth
        T = np.array(result["T_cam2stage"])
        assert abs(T[0, 0] - scale_mm_per_px) < 0.001  # 1% tolerance
        assert abs(T[1, 1] - scale_mm_per_px) < 0.001

    def test_set_get_transform(self, calibrator):
        """Test setting and getting transformation matrix."""
        T = np.array([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]])

        calibrator.set_transform(T)
        T_retrieved = calibrator.get_transform()

        assert T_retrieved is not None
        np.testing.assert_array_almost_equal(T_retrieved, T)

        # Test that retrieved is a copy
        T_retrieved[0, 0] = 999.0
        assert calibrator.get_transform()[0, 0] != 999.0

    def test_pixel_to_stage_delta(self, calibrator):
        """Test pixel displacement to stage displacement conversion."""
        # Set transformation
        T = np.array([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]])  # 20 µm/px
        calibrator.set_transform(T)

        # Test pure horizontal displacement
        dx_um, dy_um = calibrator.pixel_to_stage_delta(100.0, 0.0)
        assert abs(dx_um - 2000.0) < 1e-6  # 100 px * 20 µm/px = 2000 µm
        assert abs(dy_um - 0.0) < 1e-6

        # Test pure vertical displacement
        dx_um, dy_um = calibrator.pixel_to_stage_delta(0.0, 50.0)
        assert abs(dx_um - 0.0) < 1e-6
        assert abs(dy_um - 1000.0) < 1e-6  # 50 px * 20 µm/px = 1000 µm

        # Test diagonal displacement
        dx_um, dy_um = calibrator.pixel_to_stage_delta(100.0, 50.0)
        assert abs(dx_um - 2000.0) < 1e-6
        assert abs(dy_um - 1000.0) < 1e-6

    def test_pixel_to_stage_delta_uncalibrated(self, calibrator):
        """Test pixel to stage conversion without calibration."""
        with pytest.raises(RuntimeError, match="not calibrated"):
            calibrator.pixel_to_stage_delta(10.0, 10.0)

    def test_get_current_tag(self, calibrator):
        """Test finding tag closest to ROI center."""
        # Create synthetic image with tags
        img = np.zeros((1000, 1000), dtype=np.uint8)

        # Mock detect_tags to return predefined tags
        def mock_detect(img):
            return {
                50: (200.0, 300.0),  # Far from center
                51: (500.0, 500.0),  # At center
                52: (800.0, 700.0)   # Far from center
            }

        # Temporarily replace detect method
        original_detect = calibrator.detect_tags
        calibrator.detect_tags = mock_detect

        try:
            # Test with default ROI center (image center = 500, 500)
            tag_info = calibrator.get_current_tag(img)
            assert tag_info is not None
            assert tag_info[0] == 51  # Tag 51 is closest to center
            assert tag_info[1] == 500.0
            assert tag_info[2] == 500.0

            # Test with custom ROI center
            tag_info = calibrator.get_current_tag(img, roi_center=(200.0, 300.0))
            assert tag_info is not None
            assert tag_info[0] == 50  # Tag 50 is now closest

        finally:
            # Restore original method
            calibrator.detect_tags = original_detect


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
