import json
import time
import numpy as np

from imswitch.imcommon.framework import Signal, SignalInterface

# Lazy import to avoid circular import issues
# The import chain imcommon.model -> SharedAttributes -> imcontrol.model.metadata
# -> imcontrol.model.managers -> initLogger -> imcommon.model creates a cycle
HAS_METADATA_HUB = False
SharedAttrValue = None

def _get_shared_attr_value_class():
    """Lazy import of SharedAttrValue to avoid circular imports."""
    global HAS_METADATA_HUB, SharedAttrValue
    if SharedAttrValue is None:
        try:
            from imswitch.imcontrol.model.metadata.schema import SharedAttrValue as _SharedAttrValue
            SharedAttrValue = _SharedAttrValue
            HAS_METADATA_HUB = True
        except ImportError:
            HAS_METADATA_HUB = False
    return SharedAttrValue


def _is_shared_attr_value(value):
    """Check if value is a SharedAttrValue instance (with lazy import)."""
    cls = _get_shared_attr_value_class()
    return cls is not None and isinstance(value, cls)


class SharedAttributes(SignalInterface):
    sigAttributeSet = Signal(object, object)  # (key, value)

    def __init__(self):
        super().__init__()
        self._data = {}

    def getSharedAttributes(self):
        """ 
        Returns a dictionary of HDF5 attributes representing this object.
        
        If values are SharedAttrValue objects, extracts the actual value.
        Also includes metadata as separate keys if available.
        """
        attrs = {}
        for key, value in self._data.items():
            key_str = ':'.join(key)
            
            # Check if value is a SharedAttrValue
            if _is_shared_attr_value(value):
                attrs[key_str] = value.value
                # Add metadata as separate keys
                if value.units:
                    attrs[f"{key_str}:units"] = value.units
                if value.timestamp:
                    attrs[f"{key_str}:timestamp"] = value.timestamp
                if value.source:
                    attrs[f"{key_str}:source"] = value.source
            else:
                attrs[key_str] = value

        return attrs

    def getJSON(self):
        """ 
        Returns a JSON representation of this instance.
        
        If values are SharedAttrValue objects, includes full metadata.
        """
        attrs = {}
        for key, value in self._data.items():
            parent = attrs
            for i in range(len(key) - 1):
                if key[i] not in parent:
                    parent[key[i]] = {}
                parent = parent[key[i]]

            # Check if value is a SharedAttrValue
            if _is_shared_attr_value(value):
                parent[key[-1]] = {
                    'value': value.value,
                    'timestamp': value.timestamp,
                    'units': value.units,
                    'dtype': value.dtype,
                    'source': value.source,
                    'valid': value.valid,
                }
            else:
                parent[key[-1]] = value

        # Custom serializer for numpy types and other special objects
        def json_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return str(obj)

        return json.dumps(attrs, default=json_serializer)

    def update(self, data):
        """ Updates this object with the data in the given dictionary or
        SharedAttributes object. """
        if isinstance(data, SharedAttributes):
            data = data._data

        for key, value in data.items():
            self[key] = value

    def __getitem__(self, key):
        self._validateKey(key)
        value = self._data[key]
        # For backwards compatibility, return raw value if it's a SharedAttrValue
        if _is_shared_attr_value(value):
            return value.value
        return value
    
    def get_typed(self, key):
        """
        Get the full typed value (SharedAttrValue) if available.
        
        Returns:
            SharedAttrValue if available, otherwise raw value
        """
        self._validateKey(key)
        return self._data.get(key)

    def __setitem__(self, key, value):
        self._validateKey(key)
        # Store the value as-is (can be raw value or SharedAttrValue)
        self._data[key] = value
        # For signal emission, unwrap SharedAttrValue to maintain backwards compatibility
        if _is_shared_attr_value(value):
            self.sigAttributeSet.emit(key, value.value)
        else:
            self.sigAttributeSet.emit(key, value)

    def __iter__(self):
        yield from self._data.items()

    @classmethod
    def fromHDF5File(cls, file, dataset):
        """ Loads the attributes from a HDF5 file into a SharedAttributes
        object. """
        attrs = cls()
        for key, value in file[dataset].attrs.items():
            keyTuple = tuple(key.split(':'))
            attrs[keyTuple] = value
        return attrs

    @staticmethod
    def _validateKey(key):
        if type(key) is not tuple:
            raise TypeError('Key must be a tuple of strings')

        for keySegment in key:
            if not isinstance(keySegment, str):
                raise TypeError('Key must be a tuple of strings')

            if ':' in keySegment:
                raise KeyError('Key must not contain ":"')


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
