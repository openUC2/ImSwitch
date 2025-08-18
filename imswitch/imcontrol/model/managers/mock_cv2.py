"""Mock cv2 module for testing when OpenCV is not available."""

import sys


class MockCV2:
    """Mock cv2 module to allow imports when opencv-python is not installed."""
    
    # Common constants
    INTER_LINEAR = 1
    INTER_CUBIC = 2
    INTER_NEAREST = 0
    COLOR_BGR2RGB = 4
    COLOR_RGB2BGR = 4
    
    def __getattr__(self, name):
        """Return a mock function for any missing attribute."""
        def mock_function(*args, **kwargs):
            return None
        return mock_function
    
    def imread(self, *args, **kwargs):
        return None
    
    def imwrite(self, *args, **kwargs):
        return True
    
    def resize(self, *args, **kwargs):
        return None
    
    def cvtColor(self, *args, **kwargs):
        return None


# Install the mock module
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = MockCV2()