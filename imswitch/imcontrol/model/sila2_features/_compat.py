"""
SiLA2 compatibility helpers for OpenUC2 ImSwitch.

When unitelabs-cdk is not installed, provides no-op stub decorators and
a fallback base class so that the feature modules can be imported without
errors. The stubs ensure that the decorated methods remain valid Python
but do nothing special at import time.
"""

import abc


def _identity_decorator(*args, **kwargs):
    """A no-op decorator factory that returns the decorated function unchanged."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Used as @decorator without parentheses
        return args[0]
    # Used as @decorator() or @decorator(name="...")
    def wrapper(fn):
        return fn
    return wrapper


try:
    from unitelabs.cdk import sila

    SilaFeatureBase = sila.Feature
    # Conditionally grab decorators that may not exist in all CDK versions
    UnobservableProperty = sila.UnobservableProperty
    UnobservableCommand = sila.UnobservableCommand
    # sila.Response was removed in newer CDK versions; fall back to a no-op
    Response = getattr(sila, "Response", _identity_decorator) # TODO: This is old behavior 
except ImportError:
    sila = None

    class SilaFeatureBase(abc.ABC):
        """Stub base class used when unitelabs-cdk is not installed."""
        def __init__(self, **kwargs):
            # Accept and discard sila.Feature kwargs (originator, category, etc.)
            pass

    UnobservableProperty = _identity_decorator
    UnobservableCommand = _identity_decorator
    #Response = _identity_decorator
