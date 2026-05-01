"""
ImSwitch :class:`PositionerManager` that drives Micro-Manager XY and Z stages
through ``pymmcore-plus``.

Recognised ``managerProperties`` keys:

==================  ========  ==========================================================
Key                 Required  Description
==================  ========  ==========================================================
cfgPath             no*       Path to a Micro-Manager ``.cfg`` file.
adapterPath         no        Override the adapter search directory.
xyAdapterName       no*       Adapter for the XY stage when loading manually.
xyDeviceName        no*       Device name for the XY stage when loading manually.
xyDeviceLabel       no        Label for the XY stage device (default ``"XY"``).
zAdapterName        no*       Adapter for the focus stage when loading manually.
zDeviceName        no*       Device name for the focus stage when loading manually.
zDeviceLabel        no        Label for the focus device (default ``"Z"``).
==================  ========  ==========================================================

\\* Either ``cfgPath`` or the relevant adapter / device pairs must be provided.

The axes exposed to ImSwitch are taken from ``positionerInfo.axes``.
Movements are issued in micrometers (Micro-Manager's native unit).
"""

from __future__ import annotations

from typing import Optional

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.managers import MMCoreManager
from .PositionerManager import PositionerManager


class MMCorePositionerManager(PositionerManager):
    """Positioner manager backed by Micro-Manager XY / Z stages."""

    def __init__(self, positionerInfo, name, **lowLevelManagers):
        self._logger = initLogger(self, instanceName=name)
        self._props = dict(positionerInfo.managerProperties or {})

        cfg_path = self._props.get("cfgPath")
        adapter_path = self._props.get("adapterPath")
        adapter_paths = [adapter_path] if adapter_path else None

        self._xy_label: Optional[str] = self._props.get("xyDeviceLabel", "XY")
        self._z_label: Optional[str] = self._props.get("zDeviceLabel", "Z")

        xy_adapter = self._props.get("xyAdapterName")
        xy_device = self._props.get("xyDeviceName")
        z_adapter = self._props.get("zAdapterName")
        z_device = self._props.get("zDeviceName")

        if cfg_path:
            self._core = MMCoreManager.ensure_loaded(cfg_path, adapter_paths)
            # Use the labels from the .cfg unless explicitly overridden.
            if "xyDeviceLabel" not in self._props:
                xy = self._core.getXYStageDevice()
                self._xy_label = xy if xy else None
            if "zDeviceLabel" not in self._props:
                z = self._core.getFocusDevice()
                self._z_label = z if z else None
        elif (xy_adapter and xy_device) or (z_adapter and z_device):
            self._core = MMCoreManager.ensure_core(adapter_paths)
            loaded = set(self._core.getLoadedDevices())
            if xy_adapter and xy_device:
                if self._xy_label not in loaded:
                    self._core.loadDevice(self._xy_label, xy_adapter, xy_device)
                    self._core.initializeDevice(self._xy_label)
            else:
                self._xy_label = None
            if z_adapter and z_device:
                if self._z_label not in loaded:
                    self._core.loadDevice(self._z_label, z_adapter, z_device)
                    self._core.initializeDevice(self._z_label)
            else:
                self._z_label = None
        else:
            raise ValueError(
                f"MMCorePositionerManager '{name}' requires either 'cfgPath' or "
                "'xy'/'z' adapter + device names in managerProperties."
            )

        if self._xy_label:
            try:
                self._core.setXYStageDevice(self._xy_label)
            except Exception:
                self._logger.warning(
                    f"Could not set XY stage to '{self._xy_label}'", exc_info=True
                )
        if self._z_label:
            try:
                self._core.setFocusDevice(self._z_label)
            except Exception:
                self._logger.warning(
                    f"Could not set focus device to '{self._z_label}'", exc_info=True
                )

        # ImSwitch always seeds initial positions to zero (positions are read
        # back lazily via getPosition).
        super().__init__(
            positionerInfo,
            name,
            initialPosition={axis: 0 for axis in positionerInfo.axes},
            initialSpeed={axis: 0 for axis in positionerInfo.axes},
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, acceleration=None, speed=None, isEnable=None, timeout=0, is_reduced=True):    
        dist = float(value)
        # For relative moves, the distance is as given. For absolute moves, we need to calculate the distance to move from the current position.
        if not is_absolute:
            pass
        else:
            current_pos = self.getPosition(axis)
            dist = dist - current_pos
        if axis in ("X", "Y") and self._xy_label:
            dx = dist if axis == "X" else 0.0
            dy = dist if axis == "Y" else 0.0
            self._core.setRelativeXYPosition(dx, dy)
            if is_blocking:
                self._core.waitForDevice(self._xy_label)
        elif axis == "Z" and self._z_label:
            self._core.setRelativePosition(dist)
            if is_blocking:
                self._core.waitForDevice(self._z_label)
        else:
            self._logger.warning(f"Ignoring move on unsupported axis '{axis}'")
            return self._position[axis]

        new_pos = self.getPosition(axis)
        self._position[axis] = new_pos
        self._commChannel.sigUpdateMotorPosition.emit() # TODO: This is a hacky workaround to force Imswitch to update the motor positions in the gui..

        return new_pos

    def setPosition(self, position, axis):
        position = float(position)
        if axis == "X" and self._xy_label:
            self._core.setXYPosition(position, self._core.getYPosition())
            self._core.waitForDevice(self._xy_label)
        elif axis == "Y" and self._xy_label:
            self._core.setXYPosition(self._core.getXPosition(), position)
            self._core.waitForDevice(self._xy_label)
        elif axis == "Z" and self._z_label:
            self._core.setPosition(position)
            self._core.waitForDevice(self._z_label)
        else:
            self._logger.warning(f"Ignoring setPosition on unsupported axis '{axis}'")
            return self._position[axis]

        self._position[axis] = position
        return position

    def getPosition(self, axis):
        try:
            if axis == "X" and self._xy_label:
                return float(self._core.getXPosition())
            if axis == "Y" and self._xy_label:
                return float(self._core.getYPosition())
            if axis == "Z" and self._z_label:
                return float(self._core.getPosition())
        except Exception:
            self._logger.warning(
                f"Failed to read MMCore position for axis '{axis}'", exc_info=True
            )
        return self._position.get(axis, 0.0)

    def moveForever(self, speed=(0, 0, 0, 0), is_stop=False):
        # Micro-Manager generic stage API has no jog primitive.
        self._logger.warning("moveForever is not supported by MMCorePositionerManager")

    def finalize(self) -> None:
        # Stages do not need explicit cleanup – the shared core unloads them.
        pass


# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
