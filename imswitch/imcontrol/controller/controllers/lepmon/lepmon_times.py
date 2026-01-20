"""
LepmonOS Time Calculation Module

This module handles time calculations for the Lepmon moth trap:
- Sunrise/sunset times based on GPS coordinates
- Experiment start/end times
- Power management times
- Moon phase information

Mirrors times.py from LepmonOS_update.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass


# Try to import astronomical libraries
try:
    from timezonefinder import TimezoneFinder
    import pytz
    import ephem
    HAS_ASTRO = True
except ImportError:
    HAS_ASTRO = False
    print("Astronomical libraries not available - using default times")


@dataclass
class SunTimes:
    """Sun time data"""
    sunset: datetime
    sunrise: datetime
    timezone: any
    timezone_name: str = ""


@dataclass
class MoonData:
    """Moon phase data"""
    moonrise: Optional[datetime] = None
    moonset: Optional[datetime] = None
    phase: float = 0.0  # 0-100%
    max_altitude: float = 0.0  # degrees


@dataclass
class ExperimentTimes:
    """Experiment timing configuration"""
    start_time: str = "18:30:00"  # HH:MM:SS
    end_time: str = "06:30:00"
    power_on: str = ""  # Full datetime string
    power_off: str = ""


class LepmonTimes:
    """
    Time calculator for Lepmon moth trap.
    
    Calculates:
    - Sunrise/sunset based on GPS coordinates
    - Experiment start/end times (relative to sunset/sunrise)
    - Power management times for ATtiny controller
    - Moon phase information
    
    This mirrors times.py from LepmonOS_update.
    """
    
    def __init__(self, 
                 latitude: float = 50.9271,
                 longitude: float = 11.5892,
                 minutes_to_sunset: int = 15,
                 minutes_to_sunrise: int = 60):
        """
        Initialize time calculator.
        
        Args:
            latitude: GPS latitude
            longitude: GPS longitude
            minutes_to_sunset: Minutes before sunset to start experiment
            minutes_to_sunrise: Minutes before sunrise to end experiment
        """
        self.latitude = latitude
        self.longitude = longitude
        self.minutes_to_sunset = minutes_to_sunset
        self.minutes_to_sunrise = minutes_to_sunrise

        
        # Cached values
        self._sun_times: Optional[SunTimes] = None
        self._last_sun_calculation: Optional[datetime] = None
        self._timezone = None
    
    def update_location(self, latitude: float, longitude: float):
        """Update GPS coordinates"""
        self.latitude = latitude
        self.longitude = longitude
        self._sun_times = None  # Clear cache
        
    def set_coordinates(self, latitude: float, longitude: float):
        """Update GPS coordinates"""
        self.latitude = latitude
        self.longitude = longitude
        self._sun_times = None  # Clear cache
    
    def set_time_offsets(self, minutes_to_sunset: int, minutes_to_sunrise: int):
        """Update time offset configuration"""
        self.minutes_to_sunset = minutes_to_sunset
        self.minutes_to_sunrise = minutes_to_sunrise
    
    # ---------------------- Timezone ---------------------- #
    
    def get_timezone(self):
        """
        Get timezone for current coordinates.
        
        Equivalent to LepmonOS berechne_zeitzone().
        
        Returns:
            pytz timezone object or None
        """
        if not HAS_ASTRO:
            return None
        
        try:
            tf = TimezoneFinder()
            timezone_name = tf.timezone_at(lat=self.latitude, lng=self.longitude)
            
            if timezone_name is None:
                print("No timezone found for coordinates")
                return None
            
            self._timezone = pytz.timezone(timezone_name)
            return self._timezone
            
        except Exception as e:
            print(f"Timezone calculation failed: {e}")
            return None
    
    # ---------------------- Sun Times ---------------------- #
    
    def get_sun_times(self, date: Optional[datetime] = None) -> SunTimes:
        """
        Calculate sunset and sunrise times for given date.
        
        Equivalent to LepmonOS get_sun().
        
        Args:
            date: Date to calculate for (default: today)
            
        Returns:
            SunTimes dataclass with sunset, sunrise, and timezone
        """
        if not HAS_ASTRO:
            # Return default times if libraries not available
            now = datetime.now()
            sunset = now.replace(hour=18, minute=30, second=0, microsecond=0)
            sunrise = (now + timedelta(days=1)).replace(hour=6, minute=30, second=0, microsecond=0)
            return SunTimes(sunset=sunset, sunrise=sunrise, timezone=None)
        
        try:
            day = date or datetime.utcnow()
            
            # Get timezone
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lat=self.latitude, lng=self.longitude)
            tz = pytz.timezone(timezone_str) if timezone_str else pytz.UTC
            
            # Create observer
            obs = ephem.Observer()
            obs.lat = str(self.latitude)
            obs.lon = str(self.longitude)
            obs.date = day.strftime("%Y/%m/%d")  # UTC date
            
            # Calculate sun times
            sunrise_utc = obs.next_rising(ephem.Sun(), use_center=True).datetime()
            sunset_utc = obs.next_setting(ephem.Sun(), use_center=True).datetime()
            
            # Convert to local time
            sunrise_local = pytz.utc.localize(sunrise_utc).astimezone(tz)
            sunset_local = pytz.utc.localize(sunset_utc).astimezone(tz)
            
            # Ensure sunrise is after sunset (next day)
            if sunrise_local.date() == sunset_local.date():
                sunrise_local = sunrise_local + timedelta(days=1)
            
            self._sun_times = SunTimes(
                sunset=sunset_local,
                sunrise=sunrise_local,
                timezone=tz,
                timezone_name=timezone_str or ""
            )
            self._last_sun_calculation = datetime.now()
            
            return self._sun_times
            
        except Exception as e:
            print(f"Sun time calculation failed: {e}")
            # Return default times
            now = datetime.now()
            sunset = now.replace(hour=18, minute=30, second=0, microsecond=0)
            sunrise = (now + timedelta(days=1)).replace(hour=6, minute=30, second=0, microsecond=0)
            return SunTimes(sunset=sunset, sunrise=sunrise, timezone=None)
    
    # ---------------------- Moon Data ---------------------- #
    
    def get_moon_data(self) -> MoonData:
        """
        Calculate moon phase and times.
        
        Equivalent to LepmonOS get_moon().
        
        Returns:
            MoonData dataclass with moon information
        """
        if not HAS_ASTRO:
            return MoonData()
        
        try:
            tz = self.get_timezone() or pytz.UTC
            now_local = datetime.now()
            now_utc = now_local.astimezone(pytz.utc) if now_local.tzinfo else pytz.utc.localize(now_local)
            
            observer = ephem.Observer()
            observer.lat = str(self.latitude)
            observer.lon = str(self.longitude)
            
            # Calculate moon times
            moonrise = ephem.localtime(observer.previous_rising(ephem.Moon(), start=now_utc))
            moonset = ephem.localtime(observer.next_setting(ephem.Moon(), start=now_utc))
            
            # Handle edge case when moon is up all night
            if moonset and moonrise and (moonset - moonrise).total_seconds() / 3600 > 13:
                moonset = ephem.localtime(observer.previous_setting(ephem.Moon(), start=now_utc))
                moonrise = ephem.localtime(observer.next_rising(ephem.Moon(), start=now_utc))
            
            # Calculate moon phase
            moon = ephem.Moon(now_utc)
            phase = moon.phase  # 0-100%
            
            # Calculate max altitude
            next_transit = observer.next_transit(moon, start=now_utc)
            observer.date = next_transit
            moon.compute(observer)
            max_altitude = moon.alt * 180.0 / ephem.pi  # Convert to degrees
            
            return MoonData(
                moonrise=moonrise,
                moonset=moonset,
                phase=phase,
                max_altitude=max_altitude
            )
            
        except Exception as e:
            print(f"Moon calculation failed: {e}")
            return MoonData()
    
    # ---------------------- Experiment Times ---------------------- #
    
    def get_experiment_times(self) -> ExperimentTimes:
        """
        Calculate experiment start and end times.
        
        Equivalent to LepmonOS get_experiment_times().
        
        Experiment starts: sunset - minutes_to_sunset
        Experiment ends: sunrise - minutes_to_sunrise
        
        Returns:
            ExperimentTimes dataclass
        """
        sun = self.get_sun_times()
        
        # Calculate experiment times
        exp_start = sun.sunset - timedelta(minutes=self.minutes_to_sunset)
        exp_end = sun.sunrise - timedelta(minutes=self.minutes_to_sunrise)
        
        # Remove timezone for string formatting
        exp_start_naive = exp_start.replace(tzinfo=None) if exp_start.tzinfo else exp_start
        exp_end_naive = exp_end.replace(tzinfo=None) if exp_end.tzinfo else exp_end
        
        return ExperimentTimes(
            start_time=exp_start_naive.strftime("%H:%M:%S"),
            end_time=exp_end_naive.strftime("%H:%M:%S"),
        )
    
    def get_power_times(self) -> Tuple[str, str]:
        """
        Calculate power management times for ATtiny controller.
        
        Equivalent to LepmonOS get_times_power().
        
        Power on: sunset - minutes_to_sunset - 15min (buffer for startup)
        Power off: sunrise - 55min
        
        Returns:
            Tuple of (power_on_datetime_str, power_off_datetime_str)
        """
        sun = self.get_sun_times()
        
        # Calculate power times
        power_on = sun.sunset - timedelta(minutes=self.minutes_to_sunset) - timedelta(minutes=15)
        power_off = sun.sunrise - timedelta(minutes=55)
        
        # Remove timezone and format
        power_on_naive = power_on.replace(tzinfo=None) if power_on.tzinfo else power_on
        power_off_naive = power_off.replace(tzinfo=None) if power_off.tzinfo else power_off
        
        return (
            power_on_naive.strftime('%Y-%m-%d %H:%M:%S'),
            power_off_naive.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    # ---------------------- Time Checks ---------------------- #
    
    def is_in_capture_window(self, current_time: Optional[str] = None) -> bool:
        """
        Check if current time is within the capture window.
        
        Args:
            current_time: Time string HH:MM:SS (default: now)
            
        Returns:
            True if within capture window
        """
        exp_times = self.get_experiment_times()
        
        if current_time is None:
            current_time = datetime.now().strftime("%H:%M:%S")
        
        return self._is_in_time_range(exp_times.start_time, exp_times.end_time, current_time)
    
    def _is_in_time_range(self, start_time: str, end_time: str, current_time: str) -> bool:
        """
        Check if current time is within a time range (handles midnight crossing).
        
        Args:
            start_time: Start time HH:MM:SS
            end_time: End time HH:MM:SS  
            current_time: Current time HH:MM:SS
            
        Returns:
            True if current_time is between start and end
        """
        try:
            start = datetime.strptime(start_time, "%H:%M:%S")
            end = datetime.strptime(end_time, "%H:%M:%S")
            current = datetime.strptime(current_time, "%H:%M:%S")
            
            # Handle midnight crossing
            if end < start:
                # Range crosses midnight
                return current >= start or current <= end
            else:
                return start <= current <= end
                
        except ValueError:
            return False
    
    def is_after_lepiled_end(self, current_time: str, lepiled_end_offset_hours: int = 6) -> bool:
        """
        Check if current time is after LepiLED should turn off.
        
        LepiLED typically turns off after a set number of hours after sunset.
        
        Args:
            current_time: Current time HH:MM:SS
            lepiled_end_offset_hours: Hours after sunset to end LepiLED
            
        Returns:
            True if LepiLED should be off
        """
        sun = self.get_sun_times()
        lepiled_end = sun.sunset + timedelta(hours=lepiled_end_offset_hours)
        lepiled_end_str = lepiled_end.strftime("%H:%M:%S") if lepiled_end.tzinfo is None else \
                          lepiled_end.replace(tzinfo=None).strftime("%H:%M:%S")
        
        try:
            current = datetime.strptime(current_time, "%H:%M:%S")
            end = datetime.strptime(lepiled_end_str, "%H:%M:%S")
            return current >= end
        except ValueError:
            return False
    
    def is_first_hour_of_capture(self, current_time: str) -> bool:
        """
        Check if within first hour of capture window.
        
        Used for special handling during initial capture phase.
        
        Args:
            current_time: Current time HH:MM:SS
            
        Returns:
            True if in first hour
        """
        exp_times = self.get_experiment_times()
        
        try:
            start = datetime.strptime(exp_times.start_time, "%H:%M:%S")
            current = datetime.strptime(current_time, "%H:%M:%S")
            first_hour_end = start + timedelta(hours=1)
            
            return start <= current < first_hour_end
        except ValueError:
            return False
    
    # ---------------------- Daylight Saving ---------------------- #
    
    def check_daylight_saving(self, date_str: str) -> Tuple[bool, int]:
        """
        Check if daylight saving time change occurs on given date.
        
        Equivalent to LepmonOS zeitumstellung_info().
        
        Args:
            date_str: Date string YYYY-MM-DD HH:MM:SS
            
        Returns:
            Tuple of (is_dst_change, offset_change)
            offset_change: -1 for spring forward, +1 for fall back
        """
        if not HAS_ASTRO:
            return False, 0
        
        tz = self.get_timezone()
        if tz is None:
            return False, 0
        
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            
            # Check offset at 2am and 3am
            dt_before = tz.localize(datetime(dt.year, dt.month, dt.day, 1, 59, 59), is_dst=None)
            dt_after = tz.localize(datetime(dt.year, dt.month, dt.day, 3, 0, 0), is_dst=None)
            
            offset_before = dt_before.utcoffset()
            offset_after = dt_after.utcoffset()
            
            if offset_before != offset_after:
                if offset_after > offset_before:
                    # Spring forward (summer time starts)
                    return True, -1
                else:
                    # Fall back (winter time starts)
                    return True, 1
            
            return False, 0
            
        except Exception as e:
            print(f"Daylight saving check failed: {e}")
            return False, 0
    
    # ---------------------- Summary ---------------------- #
    
    def get_timing_summary(self) -> dict:
        """
        Get complete timing summary for frontend.
        
        Returns:
            Dictionary with all timing information
        """
        sun = self.get_sun_times()
        exp = self.get_experiment_times()
        power_on, power_off = self.get_power_times()
        moon = self.get_moon_data()
        
        return {
            "coordinates": {
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "timezone": sun.timezone_name,
            "sun": {
                "sunset": sun.sunset.strftime("%H:%M:%S") if sun.sunset else None,
                "sunrise": sun.sunrise.strftime("%H:%M:%S") if sun.sunrise else None,
            },
            "experiment": {
                "start_time": exp.start_time,
                "end_time": exp.end_time,
                "minutes_to_sunset": self.minutes_to_sunset,
                "minutes_to_sunrise": self.minutes_to_sunrise,
            },
            "power": {
                "power_on": power_on,
                "power_off": power_off,
            },
            "moon": {
                "phase": moon.phase,
                "moonrise": moon.moonrise.strftime("%H:%M:%S") if moon.moonrise else None,
                "moonset": moon.moonset.strftime("%H:%M:%S") if moon.moonset else None,
                "max_altitude": moon.max_altitude,
            },
            "is_capture_window": self.is_in_capture_window(),
        }
