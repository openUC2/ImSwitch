"""
Central mock system for ImSwitch testing.
This module provides mocks for optional dependencies to allow tests to run
without requiring all production dependencies.
"""

import sys


class MockModule:
    """Generic mock module that can be used for any missing module."""
    
    def __init__(self, name=None):
        self._name = name or 'MockModule'
        # Add some common mock functions and classes
        self.trans = lambda x: x  # For napari.utils.translations.trans
        self.Color = MockModule  # For vispy.color.Color
        self.Compound = MockModule  # For vispy.scene.visuals.Compound
        self.Line = MockModule  # For vispy.scene.visuals.Line
        self.Markers = MockModule  # For vispy.scene.visuals.Markers
    
    def __getattr__(self, name):
        """Return a mock object for any missing attribute."""
        if name == 'trans':
            return lambda x: x
        elif name in ['Color', 'Compound', 'Line', 'Markers']:
            return MockModule
        elif name == 'APIExport':
            # Return a mock decorator that accepts arguments
            return MockDecorator
        return MockModule(f"{self._name}.{name}")
    
    def __call__(self, *args, **kwargs):
        """Allow the mock to be called like a function."""
        return MockModule(f"{self._name}()")
    
    def __iter__(self):
        """Allow iteration."""
        return iter([])
    
    def __bool__(self):
        """Return False for boolean checks."""
        return False
    
    def __mro_entries__(self, bases):
        """Required for classes that might be used as base classes."""
        return (object,)  # Return object as fallback base class
    
    def __getitem__(self, key):
        """Make the mock subscriptable for array-like operations."""
        return MockModule(f"{self._name}[{key}]")
    
    def __str__(self):
        return f"<MockModule: {self._name}>"
    
    def __repr__(self):
        return self.__str__()


class MockDecorator:
    """Mock decorator class that can handle arguments and be used as a decorator."""
    
    def __init__(self, *args, **kwargs):
        pass  # Accept any arguments
    
    def __call__(self, func):
        """Allow the mock to be used as a decorator."""
        return func  # Just return the original function unchanged


def install_all_mocks():
    """Install mock modules for all optional dependencies used in ImSwitch."""
    mock_modules = [
        # Data acquisition
        'nidaqmx',
        'nidaqmx._lib', 
        'nidaqmx.constants',
        'nidaqmx.system',
        'pyDAQmx',
        
        # Instrument control
        'pyvisa',
        'pyvisa-py',
        'visa',
        'lantz',
        'lantzdev',
        'microscope',
        
        # Communication
        'pyro5',
        'pyrpc', 
        'zmq',
        'zmq.asyncio',
        'zmq.auth',
        'zmq.backend',
        'serial',
        'pyserial',
        
        # GUI and visualization
        'napari',
        'napari.layers',
        'napari.utils',
        'napari.utils.translations',
        'napari.viewer',
        'vispy',
        'vispy.color',
        'vispy.scene',
        'vispy.scene.visuals',
        'vispy.visuals',
        'vispy.visuals.transforms',
        'pyqtgraph',
        'pyqtgraph.dockarea',
        'pyqtgraph.console', 
        'pyqtgraph.Qt',
        'pyqtgraph.parametertree',
        'pyqtgraph.widgets',
        'pyqtgraph.opengl',
        'QScintilla',
        'PyQtWebEngine',
        
        # Image processing (already installed via mock_cv2.py but adding for completeness)
        'cv2',
        
        # Scientific computing that might be missing
        'dask',
        'dask.array',
        'zarr',
        'numcodecs',
        
        # Specialized libraries
        'luddite',
        'colour',
        'colour_science',
        'piexif',
        'aiortc',
        'aiohttp',
        'numba',
        'numba.types',
        
        # Hardware specific
        'RPi',
        'RPi.GPIO',
        'luma',
        'luma.oled',
        'smbus',
        'smbus2',
    ]
    
    for module_name in mock_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MockModule(module_name)


# Install all mocks when this module is imported
install_all_mocks()