"""A stand-in for ESP32Manager on the virtual microscope.

Code that talks to the UC2 board goes through ``rs232sManager["ESP32"]._esp32``
(UC2ConfigManager, TemperatureController, I2CSensorController, the instrument
metadata). This manager puts a simulated uc2rest client there, so those paths
work without hardware. It mirrors the uc2rest method names and return shapes.

Simulated:
- temperature: a heater with an ideal controller. The reading approaches the
  target (or ambient when off) with a first-order lag, plus sensor noise.
- digital outputs 1..3: latched; -1 is a pulse (counted, ends low).
- digital inputs 1..3: a loopback of an output, a time schedule, or set by a
  simulated firmware message sent through ``serial.writeSerial``:
  ``{"task": "/digitalin_sim", "digitalinid": 1, "digitalinval": 1}``.
  ``{"task": "/estop_sim", "active": 1}`` raises the emergency stop.
- I2C sensors (SHT45 temperature/humidity, TSL2591 light), fan, bus power.

Not modelled: CAN, collision detector, PTZ, galvo. Their getters return empty
results; their setters raise NotImplementedError, which the callers log.

Setup JSON::

    "rs232devices": {"ESP32": {"managerName": "VirtualESP32Manager",
        "managerProperties": {"ambientC": 22, "heaterTauS": 60, "heaterMaxRiseC": 30,
                              "noiseC": 0.02, "humidityPct": 45, "lux": 120,
                              "digitalIn": {"1": {"mirrorOut": 1},
                                            "2": {"schedule": [[30, 1], [60, 0]]}}}}}
"""
import json
import math
import random
import threading
import time

from imswitch.imcommon.model import initLogger

N_DIGITAL = 3


class _Temperature:
    def __init__(self, client, ambient_c, tau_s, max_rise_c, noise_c):
        self._client = client
        self.ambient_c = float(ambient_c)
        self.tau_s = max(float(tau_s), 1e-3)
        self.max_rise_c = float(max_rise_c)
        self.noise_c = float(noise_c)
        self.active = False
        self.target_c = self.ambient_c
        self.gains = {"Kp": None, "Ki": None, "Kd": None}
        self._true_c = self.ambient_c
        self._t = client.clock()
        self._lock = threading.Lock()

    def _advance(self):
        now = self._client.clock()
        dt = max(0.0, now - self._t)
        self._t = now
        goal = self.ambient_c
        if self.active:
            goal = min(max(self.target_c, self.ambient_c), self.ambient_c + self.max_rise_c)
        self._true_c = goal + (self._true_c - goal) * math.exp(-dt / self.tau_s)
        return self._true_c

    def true_temperature(self):
        with self._lock:
            return self._advance()

    # uc2rest API
    def set_temperature(self, active=1, Kp=1000, Ki=0.1, Kd=0.1, target=37,
                        timeout=600000000, updaterate=1000):
        with self._lock:
            self._advance()
            self.active = bool(active)
            self.target_c = float(target)
            self.gains = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
        return {"heat": {"active": int(self.active), "target": self.target_c}}

    def stop_heating(self):
        return self.set_temperature(active=0, target=self.target_c)

    def get_temperature(self, timeout=0.5, isBlocking=False):
        return round(self.true_temperature() + random.gauss(0.0, self.noise_c), 3)

    def start_temperature_polling(self, period=5):
        return None

    def stop_temperature_polling(self):
        return None


class _DigitalOut:
    def __init__(self):
        self.values = {i: 0 for i in range(1, N_DIGITAL + 1)}
        self.pins = {i: None for i in range(1, N_DIGITAL + 1)}
        self.pulses = {i: 0 for i in range(1, N_DIGITAL + 1)}
        self.trigger_table = {}
        self.triggers_sent = []

    def reset(self):
        self.__init__()

    def _check(self, digitaloutid):
        # The UC2Config endpoints are untyped, so ids arrive as query strings.
        digitaloutid = int(digitaloutid)
        if digitaloutid not in self.values:
            raise ValueError(f"digital output {digitaloutid} does not exist (1..{N_DIGITAL})")
        return digitaloutid

    def setup_digitaloutpin(self, id=1, pin=4):
        id = self._check(id)
        self.pins[id] = pin
        return {"digitalout": {"digitaloutid": id, "digitaloutpin": pin}}

    def set_digitalout(self, digitaloutid=1, digitaloutval=0, timeout=1, is_blocking=False):
        digitaloutid = self._check(digitaloutid)
        if int(digitaloutval) == -1:
            self.pulses[digitaloutid] += 1
            self.values[digitaloutid] = 0
        else:
            self.values[digitaloutid] = 1 if int(digitaloutval) else 0
        return {"digitalout": {"digitaloutid": digitaloutid,
                               "digitaloutval": self.values[digitaloutid]}, "qid": 0}

    def get_digitalout(self, digitaloutid=1, timeout=1, is_blocking=True):
        digitaloutid = self._check(digitaloutid)
        return {"digitalout": {"digitaloutid": digitaloutid,
                               "digitaloutval": self.values[digitaloutid],
                               "digitaloutpin": self.pins[digitaloutid]}, "qid": 0}

    def reset_triggertable(self):
        self.trigger_table = {}
        return {"digitalout": {"digitaloutistriggerreset": 1}}

    def set_trigger(self, trigger1=False, delayOn1=0, delayOff1=0, trigger2=False, delayOn2=0,
                    delayOff2=0, trigger3=False, delayOn3=0, delayOff3=0):
        self.trigger_table = {1: (trigger1, delayOn1, delayOff1), 2: (trigger2, delayOn2, delayOff2),
                              3: (trigger3, delayOn3, delayOff3)}
        return {"digitalout": {"triggers": self.trigger_table}}

    def sendTrigger(self, triggerId=0):
        self.triggers_sent.append((time.time(), triggerId))
        return {"digital": {"digitalid": triggerId, "digitalval": -1}}

    def register_callback(self, key, callbackfct):
        return None


class _DigitalIn:
    def __init__(self, client, config):
        self._client = client
        self._manual = {}
        self._callbacks = {}
        self._config = {int(k): v for k, v in (config or {}).items()}
        self._last = {i: None for i in range(1, N_DIGITAL + 1)}
        self._t0 = client.clock()

    def value(self, digitalinid):
        digitalinid = int(digitalinid)
        if digitalinid not in self._last:
            raise ValueError(f"digital input {digitalinid} does not exist (1..{N_DIGITAL})")
        if digitalinid in self._manual:
            return self._manual[digitalinid]
        cfg = self._config.get(digitalinid, {})
        if "mirrorOut" in cfg:
            return self._client.digitalout.values.get(int(cfg["mirrorOut"]), 0)
        value = 0
        elapsed = self._client.clock() - self._t0
        for at_s, scheduled in sorted(cfg.get("schedule", [])):
            if elapsed >= at_s:
                value = int(scheduled)
        return value

    def set_input(self, digitalinid, value):
        """Simulate a level on an input and notify callbacks on a change."""
        digitalinid = int(digitalinid)
        self.value(digitalinid)  # validates the id
        self._manual[digitalinid] = 1 if int(value) else 0
        self.poll()

    def poll(self):
        values = [self.value(i) for i in range(1, N_DIGITAL + 1)]
        changed = any(self._last[i] is not None and self._last[i] != values[i - 1]
                      for i in range(1, N_DIGITAL + 1))
        for i in range(1, N_DIGITAL + 1):
            self._last[i] = values[i - 1]
        if changed:
            for callbacks in self._callbacks.values():
                for fct in list(callbacks):
                    try:
                        fct(list(values))
                    except Exception as e:  # mirror uc2rest: a bad callback never breaks the loop
                        print(f"Error in digitalin callback: {e}")
        return values

    # uc2rest API
    def get_digitalin(self, digitalinid=1, timeout=1, is_blocking=True):
        self.poll()
        digitalinid = int(digitalinid)
        return {"digitalin": {"digitalinid": digitalinid,
                              "digitalinval": self.value(digitalinid)}, "qid": 0}

    def act_digitalin(self, timeout=1, is_blocking=False):
        values = self.poll()
        return {"digitalin": [{"digitalinid": i + 1, "digitalinval": v}
                              for i, v in enumerate(values)], "qid": 0}

    def register_callback(self, key, callbackfct):
        self._callbacks.setdefault(key, [])
        if callbackfct not in self._callbacks[key]:
            self._callbacks[key].append(callbackfct)


class _I2C:
    def __init__(self, client, humidity_pct, lux):
        self._client = client
        self.humidity_pct = float(humidity_pct)
        self.lux = float(lux)

    def read_sht45(self, addr=0x44, node=None, precision="high", timeout=2):
        temp = self._client.temperature.true_temperature()
        return {"temperature_c": round(temp + random.gauss(0, 0.02), 3),
                "humidity_pct": round(self.humidity_pct + random.gauss(0, 0.2), 2),
                "crc_ok": True}

    def read_tsl2591(self, addr=0x29, node=None, gain=0x10, integration=0x01, timeout=2):
        lux = max(0.0, self.lux + random.gauss(0, 0.5))
        ch0 = int(lux * 40)
        ch1 = int(lux * 8)
        return {"ch0_full": ch0, "ch1_ir": ch1, "visible": ch0 - ch1, "gain": gain,
                "integration_ms": 200, "lux": round(lux, 2), "saturated": False}

    def scan(self, node=None, start=0x08, end=0x77, timeout=2):
        return [0x29, 0x44]


class _Fan:
    def __init__(self, client):
        self._client = client
        self.mode, self.wiper = "auto", 64

    def get_fan(self, timeout=1, blocking=True):
        rpm = 0 if self.mode == "off" else int(3000 * self.wiper / 127)
        return {"mode": self.mode, "wiper": self.wiper, "manual": int(self.mode == "manual"),
                "rpm": rpm, "stalled": 0, "kick": 1,
                "tempC": round(self._client.temperature.ambient_c + 8, 1), "curve": []}

    def get_temp(self, timeout=1):
        amb = self._client.temperature.ambient_c
        return {"pcb": amb + 8, "air": amb, "esp": amb + 15, "pcb_ok": 1, "air_ok": 1}

    def set_mode(self, mode="auto", wiper=None):
        if mode not in ("auto", "manual", "off"):
            raise ValueError("fan mode must be auto, manual or off")
        self.mode = mode
        if wiper is not None:
            self.wiper = int(max(0, min(127, wiper)))
        return {"fan": {"mode": self.mode, "wiper": self.wiper}}


class _State:
    def __init__(self, client):
        self._client = client
        self.power = 1
        self.estop_active = False
        self._emergency_callbacks = []

    def set_power(self, power=1, getReturn=True, timeout=1):
        self.power = 1 if int(power) else 0
        return {"state": {"power": self.power}}

    def get_power(self, timeout=1):
        return self.power

    def get_estop(self, timeout=1):
        return {"estopPolarity": 0, "estopRaw": int(self.estop_active),
                "estopActive": int(self.estop_active)}

    def is_emergency_active(self):
        return self.estop_active

    def set_emergency(self, active):
        self.estop_active = bool(active)
        if self.estop_active:
            self.power = 0
        for fct in list(self._emergency_callbacks):
            try:
                fct({"active": int(self.estop_active), "power": self.power})
            except Exception as e:
                print(f"Error in emergency callback: {e}")

    def register_emergency_callback(self, callbackfct):
        if callbackfct not in self._emergency_callbacks:
            self._emergency_callbacks.append(callbackfct)

    def unregister_emergency_callback(self, callbackfct):
        if callbackfct in self._emergency_callbacks:
            self._emergency_callbacks.remove(callbackfct)

    def get_firmware_info(self, timeout=3):
        return {"name": "UC2_Virtual", "version": "V2.0", "fwVersion": "virtual",
                "fwImage": "", "date": "", "author": "ImSwitch", "pindef": "virtual",
                "isMaster": False}

    def get_state(self, timeout=3):
        return {"identifier_name": "UC2_Virtual", "identifier_id": "V2.0", "pindef": "virtual"}

    def is_master(self, timeout=1):
        return False

    def isBusy(self, timeout=1):
        return False

    def espRestart(self, timeout=1):
        self._client.digitalout.reset()
        return True

    def pairBT(self, timeout=1):
        return True


class _Serial:
    """Connection stub; writeSerial also takes the simulation messages."""

    def __init__(self, client):
        self._client = client
        self.is_connected = True
        self.serialport = "virtual"
        self.firmware_info = client.state.get_firmware_info()
        self.DEBUG = False
        self._write_cb = self._read_cb = None

    def ping(self, timeout=0.5):
        return True

    def setWriteCallback(self, fct):
        self._write_cb = fct

    def setReadCallback(self, fct):
        self._read_cb = fct

    def register_callback(self, *args, **kwargs):
        return None

    def reconnect(self, *args, **kwargs):
        self.is_connected = True
        return True

    def closeSerial(self):
        self.is_connected = False

    def interruptCurrentSerialCommunication(self):
        return None

    def writeSerial(self, payload):
        if self._write_cb:
            self._write_cb(payload)
        msg = json.loads(payload) if isinstance(payload, str) else dict(payload)
        task = msg.get("task")
        if task == "/digitalin_sim":
            self._client.digitalin.set_input(int(msg.get("digitalinid", 1)),
                                             int(msg.get("digitalinval", 0)))
            reply = self._client.digitalin.get_digitalin(int(msg.get("digitalinid", 1)))
        elif task == "/estop_sim":
            self._client.state.set_emergency(bool(msg.get("active", 1)))
            reply = {"emergency": self._client.state.get_estop()}
        else:
            reply = {"error": f"the virtual ESP32 does not simulate {task!r}"}
        if self._read_cb:
            self._read_cb(json.dumps(reply))
        return reply


class _CallbackStub:
    """message / camera_trigger: accept registrations; nothing fires them."""

    nCallbacks = 10

    def __init__(self):
        self.callbacks = {}

    def register_callback(self, key, callbackfct):
        self.callbacks.setdefault(key, []).append(callbackfct)


class _NotModelled:
    """gpio (collision), ptz, motor, config: empty reads, setters refuse."""

    def __init__(self, what):
        self._what = what

    def register_collision_callback(self, *args, **kwargs):
        return None

    def register_event_callback(self, *args, **kwargs):
        return None

    def get_status(self, *args, **kwargs):
        return {}

    def get_joystick_direction(self, *args, **kwargs):
        return {}

    def get_speed_multiplier(self, *args, **kwargs):
        return {}

    def loadDefaultConfig(self, *args, **kwargs):
        return {}

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            raise NotImplementedError(f"the virtual ESP32 has no {self._what} ({name})")
        return refuse


class VirtualUC2Client:
    def __init__(self, props, clock=time.monotonic):
        self.clock = clock
        self.temperature = _Temperature(self, props.get("ambientC", 22.0),
                                        props.get("heaterTauS", 60.0),
                                        props.get("heaterMaxRiseC", 30.0),
                                        props.get("noiseC", 0.02))
        self.digitalout = _DigitalOut()
        self.digitalin = _DigitalIn(self, props.get("digitalIn"))
        self.i2c = _I2C(self, props.get("humidityPct", 45.0), props.get("lux", 120.0))
        self.fan = _Fan(self)
        self.state = _State(self)
        self.serial = _Serial(self)
        self.message = _CallbackStub()
        self.camera_trigger = _CallbackStub()
        self.gpio = _NotModelled("collision detector")
        self.ptz = _NotModelled("PTZ keyboard")
        self.motor = _NotModelled("motor settings")
        self.config = _NotModelled("config store")
        self.can = None
        self.is_connected = True

    def close(self):
        self.serial.closeSerial()


class VirtualESP32Manager:
    """rs232 device manager that provides the simulated client as ``_esp32``."""

    def __init__(self, rs232Info, name, **_lowLevelManagers):
        self.__logger = initLogger(self, instanceName=name)
        props = dict(getattr(rs232Info, "managerProperties", None) or {})
        self._esp32 = VirtualUC2Client(props)
        self.__logger.info("Virtual ESP32: temperature, digital I/O, I2C sensors, fan, bus power")

    def query(self, arg: str) -> str:
        return ""

    def write(self, arg: str) -> None:
        return None

    def finalize(self):
        self._esp32.close()
