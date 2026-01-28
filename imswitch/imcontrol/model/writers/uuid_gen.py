"""
UUID generation for image files with metadata-based content IDs.

Provides deterministic UUIDs derived from metadata hashes for
tamper-evident file identification.
"""

import hashlib
import json
import uuid
from typing import Dict, Any
from collections import OrderedDict


def canonicalize_metadata(metadata: Dict[str, Any]) -> bytes:
    """
    Convert metadata to canonical bytes representation.
    
    Uses stable sorting and formatting to ensure the same metadata
    always produces the same hash, regardless of dict ordering or
    floating point representation variations.
    
    Args:
        metadata: Metadata dictionary
    
    Returns:
        Canonical bytes representation
    """
    def normalize_value(v):
        """Normalize a value for stable hashing."""
        if isinstance(v, float):
            # Format floats to fixed precision
            return f"{v:.10e}"
        elif isinstance(v, dict):
            # Recursively sort dict keys
            return OrderedDict(sorted((k, normalize_value(val)) for k, val in v.items()))
        elif isinstance(v, (list, tuple)):
            # Normalize lists/tuples
            return [normalize_value(item) for item in v]
        else:
            # Return as-is for strings, ints, bools, None
            return v
    
    # Normalize the metadata
    normalized = normalize_value(metadata)
    
    # Convert to JSON with sorted keys
    canonical_json = json.dumps(
        normalized,
        sort_keys=True,
        separators=(',', ':'),  # No whitespace
        ensure_ascii=True
    )
    
    return canonical_json.encode('utf-8')


def compute_content_id(metadata: Dict[str, Any], namespace: str = "ImSwitch") -> str:
    """
    Compute a deterministic content ID from metadata.
    
    Uses SHA-256 hash of canonical metadata to generate a UUIDv5-like
    identifier. The same metadata always produces the same ID.
    
    Args:
        metadata: Metadata dictionary
        namespace: Namespace for UUID generation
    
    Returns:
        Content ID as hex string (e.g., 'sha256:abc123...')
    """
    canonical = canonicalize_metadata(metadata)
    
    # Compute SHA-256 hash
    hash_obj = hashlib.sha256()
    hash_obj.update(namespace.encode('utf-8'))
    hash_obj.update(b'\x00')  # Separator
    hash_obj.update(canonical)
    
    # Return as prefixed hex string
    return f"sha256:{hash_obj.hexdigest()}"


def compute_uuid5(metadata: Dict[str, Any], namespace_uuid: uuid.UUID = None) -> str:
    """
    Compute a UUIDv5 from metadata.
    
    Uses the standard UUID namespace approach for compatibility
    with systems that expect RFC 4122 UUIDs.
    
    Args:
        metadata: Metadata dictionary
        namespace_uuid: UUID namespace (defaults to DNS namespace)
    
    Returns:
        UUIDv5 as string
    """
    if namespace_uuid is None:
        # Use DNS namespace as default
        namespace_uuid = uuid.NAMESPACE_DNS
    
    canonical = canonicalize_metadata(metadata)
    
    # Generate UUIDv5
    content_uuid = uuid.uuid5(namespace_uuid, canonical.decode('utf-8'))
    
    return str(content_uuid)


def generate_session_uuid(
    detector_name: str,
    start_timestamp: float,
    user: str = None,
    project: str = None
) -> str:
    """
    Generate a session UUID from key identifying metadata.
    
    Args:
        detector_name: Detector name
        start_timestamp: Session start timestamp
        user: Optional user name
        project: Optional project name
    
    Returns:
        Session UUID as string
    """
    metadata = {
        'detector': detector_name,
        'timestamp': start_timestamp,
    }
    
    if user:
        metadata['user'] = user
    if project:
        metadata['project'] = project
    
    return compute_uuid5(metadata)


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
