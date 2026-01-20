"""
LepmonOS OLED Display Control Module

This module handles OLED display control for the Lepmon moth trap.
Mirrors OLED_panel.py and messages.py from LepmonOS_update.

Display: SH1106 128x64 OLED via I2C
Default address: 0x3C on I2C port 1
"""

import os
import time
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass


# Try to import OLED libraries
try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import sh1106
    from PIL import ImageFont, Image
    HAS_OLED = True
except ImportError:
    HAS_OLED = False
    print("OLED libraries not available - running in OLED simulation mode")


# OLED Configuration
OLED_I2C_PORT = 1
OLED_I2C_ADDRESS = 0x3C
OLED_WIDTH = 128
OLED_HEIGHT = 64


# Message register for multilingual support
# Subset of messages from LepmonOS messages.py
MESSAGE_REGISTER = {
    # Device info
    "device_1": {
        "sleep": 1,
        "de": ("Version: {hardware}", "SN: {sn}", "Firmware: {version}"),
        "en": ("Version: {hardware}", "SN: {sn}", "Firmware: {version}"),
        "es": ("Versión: {hardware}", "SN: {sn}", "Firmware: {version}")
    },
    # HMI messages
    "hmi_01": {
        "sleep": 0,
        "de": ("Menü öffnen", "bitte Enter drücken", "(linke Taste)"),
        "en": ("Menu open", "please press Enter", "(left button)"),
        "es": ("Abrir menú", "por favor presione Enter", "(botón izquierdo)")
    },
    "hmi_02": {
        "sleep": 0,
        "de": ("Menü öffnen", "bitte Enter drücken", "(rechte Taste)"),
        "en": ("Menu open", "please press Enter", "(right button)"),
        "es": ("Abrir menú", "por favor presione Enter", "(botón derecho)")
    },
    "hmi_03": {
        "sleep": 0,
        "de": ("Eingabe Menü", "geöffnet", ""),
        "en": ("Input menu", "opened", ""),
        "es": ("Menú de entrada", "abierto", "")
    },
    # Power messages
    "hmi_06": {
        "sleep": 1,
        "de": ("Stromversorgung:", "▲= Solar →= zurück", "▼= Netz"),
        "en": ("Power supply:", "▲= Solar →= back", "▼= Mains"),
        "es": ("Fuente de energía:", "▲= Solar →= atrás", "▼= Red")
    },
    "hmi_07": {
        "sleep": 2,
        "de": ("Stromversorgung", "Solar", ""),
        "en": ("Power supply", "Solar", ""),
        "es": ("Fuente de energía", "Solar", "")
    },
    "hmi_08": {
        "sleep": 2,
        "de": ("Stromversorgung", "Netz", ""),
        "en": ("Power supply", "Mains", ""),
        "es": ("Fuente de energía", "Red", "")
    },
    # USB messages
    "hmi_09": {
        "sleep": 1,
        "de": ("USB Daten löschen?", "▲ = ja     → = zurück", "▼ = nein"),
        "en": ("Erase USB data?", "▲ = yes     → = back", "▼ = no"),
        "es": ("¿Borrar datos USB?", "▲ = sí     → = atrás", "▼ = no")
    },
    "hmi_10": {
        "sleep": 1,
        "de": ("USB Stick wird gelöscht", "bitte warten", ""),
        "en": ("Erasing USB data", "please wait", ""),
        "es": ("Borrando datos USB", "por favor espere", "")
    },
    # Heater messages
    "hmi_13": {
        "sleep": 1,
        "de": ("Scheibe heizen?", "▲ = ja     → = zurück", "▼ = nein"),
        "en": ("Heat window?", "▲ = yes     → = back", "▼ = no"),
        "es": ("¿Calentar el vidrio?", "▲ = sí     → = atrás", "▼ = no")
    },
    "hmi_14": {
        "sleep": 1,
        "de": ("Heizung aktiviert", "für 15 min", ""),
        "en": ("Heater on", "for 15 min", ""),
        "es": ("Calefacción activada", "por 15 min", "")
    },
    # Camera messages
    "cam_1": {
        "sleep": 0,
        "de": ("Kamera wird", "initialisiert", ""),
        "en": ("Camera", "initializing", ""),
        "es": ("Cámara", "inicializando", "")
    },
    "cam_2": {
        "sleep": 0,
        "de": ("Kamera bereit", "", ""),
        "en": ("Camera ready", "", ""),
        "es": ("Cámara lista", "", "")
    },
    "cam_3": {
        "sleep": 3,
        "de": ("Kamera nicht", "erkannt", ""),
        "en": ("Camera not", "detected", ""),
        "es": ("Cámara no", "detectada", "")
    },
    "cam_4": {
        "sleep": 0,
        "de": ("Fokus Modus", "aktiv", ""),
        "en": ("Focus mode", "active", ""),
        "es": ("Modo de enfoque", "activo", "")
    },
    "cam_5": {
        "sleep": 0,
        "de": ("Bildaufnahme", "läuft...", ""),
        "en": ("Capturing", "image...", ""),
        "es": ("Capturando", "imagen...", "")
    },
    "cam_6": {
        "sleep": 0,
        "de": ("Bild", "gespeichert", ""),
        "en": ("Image", "saved", ""),
        "es": ("Imagen", "guardada", "")
    },
    # Error messages
    "err_1a": {
        "sleep": 1,
        "de": ("Kamera Fehler", "Versuch: {tries}", ""),
        "en": ("Camera error", "Attempt: {tries}", ""),
        "es": ("Error de cámara", "Intento: {tries}", "")
    },
    "err_usb": {
        "sleep": 2,
        "de": ("USB Fehler", "nicht erkannt", "prüfe Anschluss"),
        "en": ("USB error", "not detected", "check connection"),
        "es": ("Error USB", "no detectado", "verifique conexión")
    },
    # Startup/shutdown
    "end_1": {
        "sleep": 1,
        "de": ("Herunterfahren in", "{time} Sekunden", ""),
        "en": ("Shutdown in", "{time} seconds", ""),
        "es": ("Apagando en", "{time} segundos", "")
    },
    "blank": {
        "sleep": 0,
        "de": ("", "", ""),
        "en": ("", "", ""),
        "es": ("", "", "")
    },
    # Capture status
    "capture_start": {
        "sleep": 1,
        "de": ("Aufnahme", "gestartet", ""),
        "en": ("Capture", "started", ""),
        "es": ("Captura", "iniciada", "")
    },
    "capture_stop": {
        "sleep": 1,
        "de": ("Aufnahme", "gestoppt", ""),
        "en": ("Capture", "stopped", ""),
        "es": ("Captura", "detenida", "")
    },
    "capture_status": {
        "sleep": 0,
        "de": ("Bilder: {count}", "Nächstes: {time}", "Frei: {space}GB"),
        "en": ("Images: {count}", "Next: {time}", "Free: {space}GB"),
        "es": ("Imágenes: {count}", "Próxima: {time}", "Libre: {space}GB")
    },
    # UV LED
    "uv_on": {
        "sleep": 0,
        "de": ("UV-LED", "eingeschaltet", ""),
        "en": ("UV LED", "on", ""),
        "es": ("LED UV", "encendido", "")
    },
    "uv_off": {
        "sleep": 0,
        "de": ("UV-LED", "ausgeschaltet", ""),
        "en": ("UV LED", "off", ""),
        "es": ("LED UV", "apagado", "")
    },
}


@dataclass
class DisplayContent:
    """Current display content"""
    line1: str = ""
    line2: str = ""
    line3: str = ""
    line4: str = ""


class LepmonOLED:
    """
    OLED display controller for Lepmon hardware.
    
    Handles:
    - Text display (3-4 lines)
    - Image display
    - Message system with multilingual support
    
    This mirrors OLED_panel.py from LepmonOS_update.
    """
    
    def __init__(self, 
                 rotate: int = 0,
                 font_path: Optional[str] = None,
                 display_callback: Optional[Callable[[str], None]] = None, 
                 language: str = "de"):
        """
        Initialize OLED controller.
        
        Args:
            rotate: Rotation in degrees (0, 90, 180, 270)
            font_path: Path to TTF font file
            display_callback: Optional callback for display updates (for WebSocket)
        """
        self.rotate = rotate
        self.display_callback = display_callback
        self.current_content = DisplayContent()
        self.language = language  # Default language
        
        # Initialize OLED device
        self.oled = None
        self.font = None
        self._initialized = False
        
        self._init_display(font_path)
    
    def _init_display(self, font_path: Optional[str] = None):
        """Initialize OLED display hardware"""
        if HAS_OLED:
            try:
                # Create I2C interface
                serial = i2c(port=OLED_I2C_PORT, address=OLED_I2C_ADDRESS)
                self.oled = sh1106(serial)
                
                # Apply rotation if needed
                if self.rotate:
                    self.oled.rotate = self.rotate // 90
                
                # Load font
                self._load_font(font_path)
                
                self._initialized = True
                print("OLED display initialized successfully")
                
            except Exception as e:
                print(f"OLED initialization failed: {e}")
                self._initialized = False
        else:
            print("Running in OLED simulation mode")
            self._load_font(font_path)
            self._initialized = True
    
    def _load_font(self, font_path: Optional[str] = None):
        """Load TrueType font for display"""
        if not HAS_OLED:
            self.font = None
            return
            
        # Try provided path first
        if font_path and os.path.exists(font_path):
            try:
                self.font = ImageFont.truetype(font_path, 14)
                return
            except Exception:
                pass
        
        # Try common font locations
        font_paths = [
            os.path.join(os.path.dirname(__file__), 'FreeSans.ttf'),
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            '/home/Ento/LepmonOS/FreeSans.ttf',
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, 14)
                    return
                except Exception:
                    pass
        
        # Fall back to default font
        try:
            self.font = ImageFont.load_default()
        except Exception:
            self.font = None
    
    def set_language(self, lang: str):
        """Set display language (de, en, es)"""
        if lang in ["de", "en", "es"]:
            self.language = lang
    
    def display_text(self, line1: str = "", line2: str = "", line3: str = "", 
                     sleeptime: float = 0):
        """
        Display text on OLED (3 lines).
        
        Equivalent to LepmonOS display_text().
        
        Args:
            line1: First line text
            line2: Second line text
            line3: Third line text
            sleeptime: Time to display in seconds
        """
        # Update internal state
        self.current_content.line1 = line1[:20]  # Max 20 chars per line
        self.current_content.line2 = line2[:20]
        self.current_content.line3 = line3[:20]
        
        # Display on OLED if available
        if HAS_OLED and self.oled and self._initialized:
            try:
                with canvas(self.oled) as draw:
                    draw.rectangle(self.oled.bounding_box, outline="black", fill="black")
                    if self.font:
                        draw.text((0, 5), line1, font=self.font, fill="white")
                        draw.text((0, 25), line2, font=self.font, fill="white")
                        draw.text((0, 45), line3, font=self.font, fill="white")
                    else:
                        draw.text((0, 5), line1, fill="white")
                        draw.text((0, 25), line2, fill="white")
                        draw.text((0, 45), line3, fill="white")
            except Exception as e:
                print(f"OLED display error: {e}")
        else:
            # Simulation mode - print to console
            print(f"[OLED] {line1}")
            print(f"[OLED] {line2}")
            print(f"[OLED] {line3}")
        
        # Call callback for WebSocket updates
        if self.display_callback:
            content = f"{line1}\n{line2}\n{line3}"
            self.display_callback(content)
        
        # Sleep if requested
        if sleeptime > 0:
            time.sleep(sleeptime)
    
    def display_text_and_image(self, line1: str, line2: str, line3: str,
                               image_path: str, sleeptime: float = 0):
        """
        Display text with image on right side.
        
        Equivalent to LepmonOS display_text_and_image().
        
        Args:
            line1-3: Text lines
            image_path: Path to 64x64 image
            sleeptime: Display duration
        """
        # Update state
        self.current_content.line1 = line1[:10]
        self.current_content.line2 = line2[:10]
        self.current_content.line3 = line3[:10]
        
        if HAS_OLED and self.oled and self._initialized:
            try:
                # Load image if exists
                logo = None
                if os.path.exists(image_path):
                    logo = Image.open(image_path).convert("1").resize((64, 64))
                
                with canvas(self.oled) as draw:
                    draw.rectangle(self.oled.bounding_box, outline="black", fill="black")
                    if logo:
                        draw.bitmap((self.oled.width - 64, 0), logo, fill=1)
                    if self.font:
                        draw.text((3, 5), line1, font=self.font, fill="white")
                        draw.text((3, 25), line2, font=self.font, fill="white")
                        draw.text((3, 45), line3, font=self.font, fill="white")
                    else:
                        draw.text((3, 5), line1, fill="white")
                        draw.text((3, 25), line2, fill="white")
                        draw.text((3, 45), line3, fill="white")
            except Exception as e:
                print(f"OLED display error: {e}")
        
        if self.display_callback:
            content = f"{line1}\n{line2}\n{line3}\n[IMG:{os.path.basename(image_path)}]"
            self.display_callback(content)
        
        if sleeptime > 0:
            time.sleep(sleeptime)
    
    def display_text_with_arrows(self, line1: str, line2: str, line3: str = "",
                                  x_position: Optional[int] = None, sleeptime: float = 0):
        """
        Display text with arrow indicators for buttons.
        
        Equivalent to LepmonOS display_text_with_arrows().
        """
        self.current_content.line1 = line1
        self.current_content.line2 = line2
        self.current_content.line3 = line3
        
        if HAS_OLED and self.oled and self._initialized:
            try:
                with canvas(self.oled) as draw:
                    draw.rectangle(self.oled.bounding_box, outline="black", fill="black")
                    if self.font:
                        draw.text((3, 5), line1, font=self.font, fill="white")
                        draw.text((3, 25), line2, font=self.font, fill="white")
                        draw.text((3, 45), line3, font=self.font, fill="white")
                        # Arrow indicators
                        draw.text((110, 5), "▲", font=self.font, fill="white")
                        draw.text((110, 25), "→", font=self.font, fill="white")
                        draw.text((110, 45), "▼", font=self.font, fill="white")
                        # Position marker
                        if x_position is not None:
                            draw.text((x_position, 38), "x", font=self.font, fill="white")
            except Exception as e:
                print(f"OLED display error: {e}")
        
        if self.display_callback:
            content = f"{line1}\n{line2}\n{line3}\n[▲→▼]"
            self.display_callback(content)
        
        if sleeptime > 0:
            time.sleep(sleeptime)
    
    def show_message(self, code: str, lang: Optional[str] = None, **values):
        """
        Display a predefined message from the message register.
        
        Equivalent to LepmonOS show_message().
        
        Args:
            code: Message code from MESSAGE_REGISTER
            lang: Language override (de, en, es)
            **values: Placeholder values for message formatting
        """
        if code not in MESSAGE_REGISTER:
            print(f"Unknown message code: {code}")
            return
        
        entry = MESSAGE_REGISTER[code]
        use_lang = lang or self.language
        
        if use_lang not in entry:
            print(f"Language {use_lang} not found for {code}, using 'de'")
            use_lang = "de"
        
        # Format lines with provided values
        lines = [line.format(**values) for line in entry[use_lang]]
        sleeptime = entry.get("sleep", 0)
        
        # Ensure 3 lines
        while len(lines) < 3:
            lines.append("")
        
        self.display_text(lines[0], lines[1], lines[2], sleeptime)
    
    def show_message_with_arrows(self, code: str, lang: Optional[str] = None,
                                  x_position: Optional[int] = None, **values):
        """Display message with arrow indicators"""
        if code not in MESSAGE_REGISTER:
            print(f"Unknown message code: {code}")
            return
        
        entry = MESSAGE_REGISTER[code]
        use_lang = lang or self.language
        
        if use_lang not in entry:
            use_lang = "de"
        
        lines = [line.format(**values) for line in entry[use_lang]]
        sleeptime = entry.get("sleep", 0)
        
        while len(lines) < 3:
            lines.append("")
        
        self.display_text_with_arrows(lines[0], lines[1], lines[2], x_position, sleeptime)
    
    def clear(self):
        """Clear the display"""
        self.display_text("", "", "", 0)
    
    def get_current_content(self) -> Dict[str, str]:
        """Get current display content"""
        return {
            "line1": self.current_content.line1,
            "line2": self.current_content.line2,
            "line3": self.current_content.line3,
            "line4": self.current_content.line4,
        }
    
    def is_available(self) -> bool:
        """Check if OLED is available"""
        return HAS_OLED and self._initialized


# Convenience function for backwards compatibility
def display_text(line1: str, line2: str, line3: str, sleeptime: float = 0):
    """Backwards compatible display function"""
    global _default_oled
    if _default_oled is None:
        _default_oled = LepmonOLED()
    _default_oled.display_text(line1, line2, line3, sleeptime)


def show_message(code: str, lang: str = "de", **values):
    """Backwards compatible show_message function"""
    global _default_oled
    if _default_oled is None:
        _default_oled = LepmonOLED()
    _default_oled.show_message(code, lang, **values)


# Default OLED instance (created on first use)
_default_oled: Optional[LepmonOLED] = None
