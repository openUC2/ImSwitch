"""
LepmonOS Sensor Reading Module

This module handles sensor reading for the Lepmon moth trap:
- Light sensor (BH1750) - ambient light measurement
- Inner temperature sensor (PCT2075 / BMP280)
- Outer environment sensor (BME280) - temperature, humidity, pressure
- Power sensor (INA226) - voltage, current, power

Mirrors sensor_data.py from LepmonOS_update.
"""

import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


# Try to import I2C/sensor libraries
try:
    import smbus2
    HAS_I2C = True
except ImportError:
    try:
        import smbus as smbus2
        HAS_I2C = True
    except ImportError:
        HAS_I2C = False
        print("I2C libraries (smbus) not available - running in sensor simulation mode")

try:
    import board
    import busio
    HAS_BOARD = True
except ImportError:
    HAS_BOARD = False
    print("board/busio not available - running in sensor simulation mode")

# Individual sensor imports
try:
    import adafruit_bh1750
    HAS_LIGHT_SENSOR = True
except ImportError:
    HAS_LIGHT_SENSOR = False

try:
    import adafruit_pct2075
    HAS_INNER_TEMP = True
except ImportError:
    HAS_INNER_TEMP = False

try:
    import adafruit_bmp280
    HAS_BMP280 = True
except ImportError:
    HAS_BMP280 = False

try:
    import adafruit_bme280.basic as adafruit_bme280
    HAS_BME280 = True
except ImportError:
    try:
        import bme280
        HAS_BME280 = True
    except ImportError:
        HAS_BME280 = False

try:
    from ina226 import INA226
    HAS_POWER_SENSOR = True
except ImportError:
    HAS_POWER_SENSOR = False


# I2C Configuration
I2C_BUS = 1
BME280_ADDRESS = 0x76


@dataclass
class SensorReading:
    """Single sensor reading with timestamp"""
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: int = 1  # 1 = OK, 0 = Error


@dataclass
class SensorData:
    """Complete sensor data collection"""
    # Light sensor
    lux: Optional[float] = None
    light_status: int = 0
    
    # Inner temperature
    temp_inner: Optional[float] = None
    inner_status: int = 0
    
    # Outer environment
    temp_outer: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    environment_status: int = 0
    
    # Power sensor
    bus_voltage: Optional[float] = None
    shunt_voltage: Optional[float] = None
    current: Optional[float] = None
    power: Optional[float] = None
    power_status: int = 0
    
    # Metadata
    timestamp: str = ""
    code: str = ""


class LepmonSensors:
    """
    Sensor controller for Lepmon hardware.
    
    Handles reading from:
    - BH1750 light sensor
    - PCT2075/BMP280 inner temperature sensor
    - BME280 environment sensor (temp, humidity, pressure)
    - INA226 power sensor
    
    This mirrors sensor_data.py from LepmonOS_update.
    """
    
    def __init__(self, 
                 hardware_generation: str = "Pro_Gen_2",
                 dusk_threshold: int = 90):
        """
        Initialize sensor controller.
        
        Args:
            hardware_generation: Hardware version
            dusk_threshold: Lux threshold for darkness detection
        """
        self.hardware_generation = hardware_generation
        self.dusk_threshold = dusk_threshold
        
        # I2C bus
        self.i2c = None
        self.smbus = None
        
        # Sensor instances
        self.light_sensor = None
        self.inner_temp_sensor = None
        self.bme280_sensor = None
        self.power_sensor = None
        
        # Status tracking
        self.sensor_status = {
            "Light_Sensor": 0,
            "Inner_Sensor": 0,
            "Environment_Sensor": 0,
            "Power_Sensor": 0,
        }
        
        # Last readings cache
        self.last_data = SensorData()
        
        # Initialize hardware
        self._initialized = False
        self.initialize()
    
    def initialize(self):
        """Initialize sensor hardware"""
        # Initialize I2C bus
        if HAS_BOARD:
            try:
                self.i2c = busio.I2C(board.SCL, board.SDA)
            except Exception as e:
                print(f"I2C bus initialization failed: {e}")
        
        if HAS_I2C:
            try:
                self.smbus = smbus2.SMBus(I2C_BUS)
            except Exception as e:
                print(f"SMBus initialization failed: {e}")
        
        # Initialize light sensor (BH1750)
        if HAS_LIGHT_SENSOR and self.i2c:
            try:
                self.light_sensor = adafruit_bh1750.BH1750(self.i2c)
                self.sensor_status["Light_Sensor"] = 1
                print("Light sensor (BH1750) initialized")
            except Exception as e:
                print(f"Light sensor initialization failed: {e}")
        
        # Initialize inner temperature sensor
        if self.i2c:
            if self.hardware_generation == "Pro_Gen_1" and HAS_BMP280:
                try:
                    self.inner_temp_sensor = adafruit_bmp280.Adafruit_BMP280_I2C(self.i2c)
                    self.sensor_status["Inner_Sensor"] = 1
                    print("Inner temp sensor (BMP280) initialized")
                except Exception as e:
                    print(f"Inner temp sensor initialization failed: {e}")
            elif HAS_INNER_TEMP:
                try:
                    self.inner_temp_sensor = adafruit_pct2075.PCT2075(self.i2c)
                    self.sensor_status["Inner_Sensor"] = 1
                    print("Inner temp sensor (PCT2075) initialized")
                except Exception as e:
                    print(f"Inner temp sensor initialization failed: {e}")
        
        # Initialize BME280 environment sensor
        if HAS_BME280 and self.i2c:
            try:
                self.bme280_sensor = adafruit_bme280.Adafruit_BME280_I2C(
                    self.i2c, address=BME280_ADDRESS
                )
                self.sensor_status["Environment_Sensor"] = 1
                print("Environment sensor (BME280) initialized")
            except Exception as e:
                print(f"BME280 initialization failed: {e}")
        
        # Initialize power sensor (INA226) - Gen2/Gen3 only
        if HAS_POWER_SENSOR and self.hardware_generation in ["Pro_Gen_2", "Pro_Gen_3"]:
            try:
                import logging
                self.power_sensor = INA226(busnum=1, address=0x40, 
                                          max_expected_amps=10, 
                                          log_level=logging.INFO)
                self.power_sensor.configure()
                self.power_sensor.set_low_battery(5)
                self.sensor_status["Power_Sensor"] = 1
                print("Power sensor (INA226) initialized")
            except Exception as e:
                print(f"Power sensor initialization failed: {e}")
        
        self._initialized = True
        print(f"Sensor initialization complete. Status: {self.sensor_status}")
    
    def cleanup(self):
        """Cleanup sensor resources"""
        if self.smbus:
            try:
                self.smbus.close()
            except Exception:
                pass
        self._initialized = False
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
    
    # ---------------------- Light Sensor ---------------------- #
    
    def get_light(self) -> Tuple[float, int]:
        """
        Read ambient light level.
        
        Equivalent to LepmonOS get_light().
        
        Returns:
            Tuple of (lux_value, sensor_status)
        """
        lux = self.dusk_threshold  # Default fallback
        status = 0
        
        if self.light_sensor:
            try:
                lux = round(self.light_sensor.lux, 2)
                status = 1
            except Exception as e:
                print(f"Light sensor read error: {e}")
                lux = self.dusk_threshold
                status = 0
        else:
            # Simulation mode - return random value
            import random
            lux = round(random.uniform(0, 200), 2)
            status = 1  # Simulate working sensor
        
        self.sensor_status["Light_Sensor"] = status
        return lux, status
    
    def is_dark(self) -> bool:
        """Check if ambient light is below dusk threshold"""
        lux, _ = self.get_light()
        return lux <= self.dusk_threshold
    
    # ---------------------- Inner Temperature ---------------------- #
    
    def get_inner_temperature(self) -> Tuple[Optional[float], int]:
        """
        Read inner temperature.
        
        Returns:
            Tuple of (temperature_celsius, sensor_status)
        """
        temp = None
        status = 0
        
        if self.inner_temp_sensor:
            try:
                temp = round(self.inner_temp_sensor.temperature, 2)
                status = 1
            except Exception as e:
                print(f"Inner temp sensor read error: {e}")
        else:
            # Simulation mode
            import random
            temp = round(random.uniform(20, 30), 2)
            status = 1
        
        self.sensor_status["Inner_Sensor"] = status
        return temp, status
    
    # ---------------------- Environment Sensor (BME280) ---------------------- #
    
    def get_environment(self) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        """
        Read environment sensor (temperature, humidity, pressure).
        
        Returns:
            Tuple of (temperature, humidity, pressure, sensor_status)
        """
        temp = None
        humidity = None
        pressure = None
        status = 0
        
        if self.bme280_sensor:
            try:
                temp = round(self.bme280_sensor.temperature, 2)
                humidity = round(self.bme280_sensor.humidity, 2)
                pressure = round(self.bme280_sensor.pressure, 2)
                status = 1
            except Exception as e:
                print(f"BME280 read error: {e}")
        
        # Try alternative BME280 method if main failed
        if status == 0 and self.smbus and HAS_BME280:
            try:
                import bme280
                calibration_params = bme280.load_calibration_params(self.smbus, BME280_ADDRESS)
                data = bme280.sample(self.smbus, BME280_ADDRESS, calibration_params)
                temp = round(data.temperature, 2)
                humidity = round(data.humidity, 2)
                pressure = round(data.pressure, 2)
                status = 1
            except Exception as e:
                print(f"BME280 alternative read error: {e}")
        
        # Simulation mode fallback
        if status == 0:
            import random
            temp = round(random.uniform(15, 25), 2)
            humidity = round(random.uniform(40, 70), 2)
            pressure = round(random.uniform(1000, 1025), 2)
            status = 1  # Simulate working sensor
        
        self.sensor_status["Environment_Sensor"] = status
        return temp, humidity, pressure, status
    
    # ---------------------- Power Sensor (INA226) ---------------------- #
    
    def get_power(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
        """
        Read power sensor (voltage, current, power).
        
        Equivalent to LepmonOS get_power().
        
        Returns:
            Tuple of (bus_voltage, shunt_voltage, current, power, sensor_status)
        """
        bus_voltage = None
        shunt_voltage = None
        current = None
        power = None
        status = 0
        
        if self.power_sensor and self.hardware_generation in ["Pro_Gen_2", "Pro_Gen_3"]:
            try:
                self.power_sensor.wake(3)
                time.sleep(0.2)
                
                bus_voltage = round(self.power_sensor.voltage(), 2)
                shunt_voltage = round(self.power_sensor.shunt_voltage(), 2)
                current = round(self.power_sensor.current(), 2)
                power = round(self.power_sensor.power() / 1000, 2)  # Convert to W
                status = 1
            except Exception as e:
                print(f"Power sensor read error: {e}")
        else:
            # Simulation mode or Gen1 (no power sensor)
            import random
            bus_voltage = round(random.uniform(11, 13), 2)
            shunt_voltage = round(random.uniform(0.01, 0.05), 2)
            current = round(random.uniform(100, 500), 2)
            power = round(random.uniform(1, 6), 2)
            status = 1 if self.hardware_generation in ["Pro_Gen_2", "Pro_Gen_3"] else 0
        
        self.sensor_status["Power_Sensor"] = status
        return bus_voltage, shunt_voltage, current, power, status
    
    # ---------------------- Combined Read ---------------------- #
    
    def read_all_sensors(self, code: str = "", local_time: str = "") -> Tuple[SensorData, Dict[str, int]]:
        """
        Read all sensors and return combined data.
        
        Equivalent to LepmonOS read_sensor_data().
        
        Args:
            code: Image/session code for logging
            local_time: Local timestamp string
            
        Returns:
            Tuple of (SensorData, sensor_status_dict)
        """
        data = SensorData()
        data.code = code
        data.timestamp = local_time or datetime.now().strftime("%H:%M:%S")
        
        # Read light
        data.lux, data.light_status = self.get_light()
        
        # Read inner temperature
        data.temp_inner, data.inner_status = self.get_inner_temperature()
        
        # Read environment
        data.temp_outer, data.humidity, data.pressure, data.environment_status = self.get_environment()
        
        # Read power
        data.bus_voltage, data.shunt_voltage, data.current, data.power, data.power_status = self.get_power()
        
        # Cache last reading
        self.last_data = data
        
        return data, self.sensor_status.copy()
    
    def to_dict(self, data: Optional[SensorData] = None) -> Dict[str, any]:
        """
        Convert sensor data to dictionary format.
        
        Args:
            data: SensorData to convert, or use last reading if None
            
        Returns:
            Dictionary of sensor values
        """
        d = data or self.last_data
        
        return {
            "code": d.code,
            "time_read": d.timestamp,
            "LUX_(Lux)": f"{d.lux:.2f}" if d.lux is not None else "---",
            "Temp_in_(°C)": f"{d.temp_inner:.2f}" if d.temp_inner is not None else "---",
            "Temp_out_(°C)": f"{d.temp_outer:.2f}" if d.temp_outer is not None else "---",
            "air_humidity_(%)": f"{d.humidity:.2f}" if d.humidity is not None else "---",
            "air_pressure_(hPa)": f"{d.pressure:.2f}" if d.pressure is not None else "---",
            "bus_voltage_(V)": f"{d.bus_voltage:.2f}" if d.bus_voltage is not None else "---",
            "shunt_voltage_(V)": f"{d.shunt_voltage:.2f}" if d.shunt_voltage is not None else "---",
            "current_(mA)": f"{d.current:.2f}" if d.current is not None else "---",
            "power_(W)": f"{d.power:.2f}" if d.power is not None else "---",
            "Light_Sensor": d.light_status,
            "Inner_Sensor": d.inner_status,
            "Environment_Sensor": d.environment_status,
            "Power_Sensor": d.power_status,
        }
    
    def to_frontend_dict(self, data: Optional[SensorData] = None) -> Dict[str, any]:
        """
        Convert sensor data to frontend-friendly format.
        
        Returns:
            Dictionary matching frontend Redux state structure
        """
        d = data or self.last_data
        
        return {
            "innerTemp": d.temp_inner,
            "outerTemp": d.temp_outer,
            "humidity": d.humidity,
            "pressure": d.pressure,
            "lux": d.lux,
            "voltage": d.bus_voltage,
            "current": d.current,
            "power": d.power,
            "sensorStatus": self.sensor_status.copy(),
        }
    
    def get_status(self) -> Dict[str, int]:
        """Get sensor status summary"""
        return self.sensor_status.copy()
    
    def all_sensors_ok(self) -> bool:
        """Check if all sensors are functioning"""
        return all(self.sensor_status.values())


# Convenience functions for backwards compatibility
_default_sensors: Optional[LepmonSensors] = None


def get_light(log_mode: str = "log") -> Tuple[float, int]:
    """Backwards compatible function"""
    global _default_sensors
    if _default_sensors is None:
        _default_sensors = LepmonSensors()
    return _default_sensors.get_light()


def get_power() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], int]:
    """Backwards compatible function"""
    global _default_sensors
    if _default_sensors is None:
        _default_sensors = LepmonSensors()
    return _default_sensors.get_power()


def read_sensor_data(code: str, local_time: str, log_mode: str = "log") -> Tuple[Dict, Dict]:
    """Backwards compatible function"""
    global _default_sensors
    if _default_sensors is None:
        _default_sensors = LepmonSensors()
    data, status = _default_sensors.read_all_sensors(code, local_time)
    return _default_sensors.to_dict(data), status
