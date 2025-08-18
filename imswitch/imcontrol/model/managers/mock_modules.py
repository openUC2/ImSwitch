"""Mock modules for testing when optional dependencies are not available."""

import sys


class MockModule:
    """Generic mock module that can be used for any missing module."""
    
    def __init__(self, name=None):
        self._name = name or 'MockModule'
    
    def __getattr__(self, name):
        """Return a mock object for any missing attribute."""
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
    
    def __str__(self):
        return f"<MockModule: {self._name}>"
    
    def __repr__(self):
        return self.__str__()


def install_mock_modules():
    """Install mock modules for common optional dependencies."""
    mock_modules = [
        'nidaqmx',
        'nidaqmx._lib',
        'nidaqmx.constants',
        'nidaqmx.system',
        'pyDAQmx', 
        'pyvisa',
        'visa',
        'lantz',
        'lantzdev',
        'microscope',
        'pyro5',
        'pyrpc',
        'zmq',
        'serial',
        'pyserial'
    ]
    
    for module_name in mock_modules:
        if module_name not in sys.modules:
            sys.modules[module_name] = MockModule(module_name)


# Install mock modules
install_mock_modules()