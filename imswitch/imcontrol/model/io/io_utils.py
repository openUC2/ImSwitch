"""
Shared utilities for I/O operations.

Contains common helper functions used across recording_service, adapters, and other I/O modules.
"""

import numpy as np
from typing import Any, Optional


def _safe_scalar_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert scalar/list-like values to float.
    
    Handles various input types including scalars, lists, tuples, numpy arrays,
    and SharedAttrValue objects that have a .value attribute.
    
    Args:
        value: The value to convert (scalar, list, array, or object with .value attr)
        default: Default value to return if conversion fails (default: None)
        
    Returns:
        Converted float value or default if conversion fails
    """
    if value is None:
        return default
    
    # Extract value from SharedAttrValue or similar wrapper objects
    if hasattr(value, 'value'):
        value = value.value
    
    # Handle list-like types
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        value = value[0]
    
    # Convert to float
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
