"""
ImSwitch :class:`LaserManager` that wraps a Micro-Manager shutter or analog
device through ``pymmcore-plus``.

Two modes are supported through the ``mode`` key in ``managerProperties``:

* ``"shutter"`` (default) – the device is treated as a binary shutter that
  is opened / closed via :py:meth:`CMMCorePlus.setShutterOpen`.
* ``"property"`` – the device exposes a numeric property (default name
  ``"Volts"``) that is written via :py:meth:`CMMCorePlus.setProperty`. This
  is the typical mode for DA-controlled lasers.

Recognised ``managerProperties`` keys:

==============  ========  ===============================================
Key             Required  Description
==============  ========  ===============================================
cfgPath         no*       Path to a Micro-Manager ``.cfg`` file.
adapterPath     no        Override the adapter search directory.
adapterName     no*       Adapter name when loading the device manually.
deviceName      no*       Device name when loading the device manually.
deviceLabel     YES       Label of the shutter / DA device.
mode            no        ``"shutter"`` (default) or ``"property"``.
propertyName    no        Property name in property mode (default ``"Volts"``).
valueUnits      no        Units string in property mode (default ``"V"``).
==============  ========  ===============================================

\\* Either ``cfgPath`` or both ``adapterName`` + ``deviceName`` must be set.
"""

from __future__ import annotations

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.managers import MMCoreManager
from .LaserManager import LaserManager


class MMCoreLaserManager(LaserManager):
    """Laser / shutter manager backed by a Micro-Manager device."""

    def __init__(self, laserInfo, name, **lowLevelManagers):
        self._logger = initLogger(self, instanceName=name)
        self._props = dict(laserInfo.managerProperties or {})

        cfg_path = self._props.get("cfgPath")
        adapter_path = self._props.get("adapterPath")
        adapter_paths = [adapter_path] if adapter_path else None
        adapter_name = self._props.get("adapterName")
        device_name = self._props.get("deviceName")

        self._label = self._props.get("deviceLabel")
        if not self._label:
            raise ValueError(
                f"MMCoreLaserManager '{name}' requires 'deviceLabel' in managerProperties."
            )

        self._mode = str(self._props.get("mode", "shutter")).lower()
        if self._mode not in ("shutter", "property"):
            raise ValueError(
                f"MMCoreLaserManager '{name}': mode must be 'shutter' or 'property' "
                f"(got '{self._mode}')."
            )
        self._property_name = self._props.get("propertyName", "Volts")

        if cfg_path:
            self._core = MMCoreManager.ensure_loaded(cfg_path, adapter_paths)
        elif adapter_name and device_name:
            self._core = MMCoreManager.ensure_core(adapter_paths)
            if self._label not in self._core.getLoadedDevices():
                self._core.loadDevice(self._label, adapter_name, device_name)
                self._core.initializeDevice(self._label)
        else:
            raise ValueError(
                f"MMCoreLaserManager '{name}' requires either 'cfgPath' or "
                "both 'adapterName' and 'deviceName'."
            )

        is_binary = self._mode == "shutter"
        value_units = "" if is_binary else self._props.get("valueUnits", "V")

        super().__init__(
            laserInfo,
            name,
            isBinary=is_binary,
            valueUnits=value_units,
            valueDecimals=0 if is_binary else 2,
        )

    def setEnabled(self, enabled: bool) -> None:
        try:
            if self._mode == "shutter":
                self._core.setShutterDevice(self._label)
                self._core.setShutterOpen(bool(enabled))
            else:
                if not enabled:
                    self._core.setProperty(self._label, self._property_name, 0)
        except Exception:
            self._logger.error(
                f"Failed to set enabled={enabled} on '{self._label}'", exc_info=True
            )

    def setValue(self, value) -> None:
        if self._mode == "property":
            try:
                self._core.setProperty(self._label, self._property_name, float(value))
            except Exception:
                self._logger.error(
                    f"Failed to set {self._label}.{self._property_name}={value}",
                    exc_info=True,
                )
        # Shutter mode is binary – nothing to write.

    def finalize(self) -> None:
        try:
            self.setEnabled(False)
        except Exception:
            pass


# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
