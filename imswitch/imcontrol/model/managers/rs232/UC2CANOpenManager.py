from imswitch.imcommon.model import initLogger

# uc2canopen is an optional dependency (only needed when a CANopen device is
# configured). Import lazily/guarded so importing this module never crashes the
# manager factory when the package is absent.
try:
    import uc2canopen  # pip install -e <repo>/UC2-REST-CANOPEN
    from uc2canopen import OD, NODE
    IS_UC2CANOPEN = True
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - exercised only when dep missing
    uc2canopen = None
    OD = None
    NODE = None
    IS_UC2CANOPEN = False
    _IMPORT_ERROR = e


class UC2CANOpenManager:
    """ Low-level wrapper around ``uc2canopen.UC2Client`` (CAN-bus transport).

    Analogous to :class:`ESP32Manager` but speaks CANopen (SDO/PDO over CAN)
    instead of JSON-over-serial/REST. It owns the shared bus connection plus the
    logical axis/device -> CAN node-id mapping, and exposes typed SDO helpers
    that the ``UC2CANOpen*`` device managers use.

    The device managers reference this manager through the ``rs232device`` key
    in their ``managerProperties`` exactly like the ESP32 managers reference
    :class:`ESP32Manager`.

    Manager properties:

    - ``interface`` -- "socketcan" (e.g. an MCP2515 HAT enumerating as ``can0``)
      or "waveshare" (USB-CAN-A serial adapter). If omitted, the client picks
      "waveshare" when ``port``/``serialport`` is given, otherwise "socketcan".
    - ``channel`` -- SocketCAN interface name (default "can0").
    - ``port`` / ``serialport`` -- serial port of the Waveshare adapter.
    - ``bitrate`` -- CAN bitrate (default 500000); must match firmware.
    - ``debug`` -- verbose transport logging.
    - ``nodeIdX/Y/Z/A`` -- override the motor node ids (default 11/12/13/14).
    - ``ledNodeId`` / ``laserNodeId`` -- override LED/laser node ids (20/21).
    """

    def __init__(self, rs232Info, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        self._settings = rs232Info.managerProperties if rs232Info is not None else {}
        self._name = name

        if not IS_UC2CANOPEN:
            raise ImportError(
                "uc2canopen is not installed in this environment. Install it with "
                "`pip install -e <path-to>/UC2-REST-CANOPEN` (pulls in python-can). "
                f"Original import error: {_IMPORT_ERROR}"
            )

        props = self._settings

        # Transport configuration
        self._interface = props.get('interface', None)
        self._channel = props.get('channel', None)
        self._port = props.get('port', props.get('serialport', None))
        self._bitrate = props.get('bitrate', 500000)
        self._debug = bool(props.get('debug', False))

        # Logical axis/device -> CAN node-id map (overridable in config).
        self.nodeIds = {
            "X": props.get('nodeIdX', NODE.MOT_X),
            "Y": props.get('nodeIdY', NODE.MOT_Y),
            "Z": props.get('nodeIdZ', NODE.MOT_Z),
            "A": props.get('nodeIdA', NODE.MOT_A),
            "LED": props.get('ledNodeId', NODE.LED),
            "LASER": props.get('laserNodeId', NODE.LASER),
        }
        # Reverse map (node id -> axis) for fast TPDO callback dispatch.
        self._axisByNode = {self.nodeIds[a]: a for a in ("X", "Y", "Z", "A")}

        self.__logger.info(
            f"Opening UC2 CANopen client (interface={self._interface}, "
            f"channel={self._channel}, port={self._port}, bitrate={self._bitrate})"
        )
        self._canDevice = uc2canopen.UC2Client(
            port=self._port,
            bitrate=self._bitrate,
            interface=self._interface,
            channel=self._channel,
            debug=self._debug,
        )

    # ------------------------------------------------------------------
    # Accessors used by the device managers
    # ------------------------------------------------------------------
    @property
    def client(self):
        """ The underlying :class:`uc2canopen.UC2Client`. """
        return self._canDevice

    def node_for_axis(self, axis):
        """ Return the CAN node id mapped to a logical axis (X/Y/Z/A/LED/LASER). """
        return self.nodeIds.get(str(axis).upper())

    def axis_for_node(self, node_id):
        """ Reverse of :meth:`node_for_axis` for motor axes (returns None if unknown). """
        return self._axisByNode.get(node_id)

    # ------------------------------------------------------------------
    # Typed SDO helpers — used for everything the high-level client doesn't
    # wrap directly (TMC params, soft limits, enable, LED pattern/brightness).
    # ``sub`` is the OD sub-index; for the single-motor-per-node layout the
    # first axis lives at sub=1.
    # ------------------------------------------------------------------
    def sdo_write(self, node_id, index, sub, value, kind="u32"):
        sdo = self._canDevice._sdo
        value = int(value)
        if kind == "u8":
            return sdo.write_u8(node_id, index, sub, value)
        if kind == "u16":
            return sdo.write_u16(node_id, index, sub, value)
        if kind == "u32":
            return sdo.write_u32(node_id, index, sub, value)
        if kind == "i32":
            return sdo.write_i32(node_id, index, sub, value)
        raise ValueError(f"Unknown SDO kind {kind!r}")

    def sdo_read(self, node_id, index, sub, kind="u32"):
        sdo = self._canDevice._sdo
        if kind == "u8":
            return sdo.read_u8(node_id, index, sub)
        if kind == "u16":
            return sdo.read_u16(node_id, index, sub)
        if kind == "u32":
            return sdo.read_u32(node_id, index, sub)
        if kind == "i32":
            return sdo.read_i32(node_id, index, sub)
        raise ValueError(f"Unknown SDO kind {kind!r}")

    def finalize(self):
        try:
            self._canDevice.close()
        except Exception as e:
            self.__logger.error(f"Error closing UC2 CANopen client: {e}")
