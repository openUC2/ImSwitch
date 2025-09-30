import subprocess
import shutil
from typing import List, Dict, Optional
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController


class WiFiController(ImConWidgetController):
    """
    Linux/Raspberry Pi only (uses `sudo nmcli`).

    Exposes:
      - scanNetworks(ifname=None) -> {"networks":[...], "ifname": "..."}
      - getAvailableNetworks() -> [...]
      - connectNetwork(ssid, password=None, ifname=None) -> {"status": ..., ...}
      - getCurrentSSID(ifname=None) -> {"ssid": "...", "ifname": "..."}
      - startAccessPoint(ssid, password, ifname=None, con_name="imswitch-hotspot", band="bg", channel=None)
      - stopAccessPoint(con_name="imswitch-hotspot")  # down+delete
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._last_scan: List[Dict] = []
        self._sudo = shutil.which("sudo") or "sudo"
        self._nmcli = shutil.which("nmcli") or "nmcli"

    # ---------- helpers ----------

    def _run(self, args: List[str]) -> subprocess.CompletedProcess:
        cmd = [self._sudo, "-n", self._nmcli] + args
        self._logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        
        # Check for NetworkManager connection issues
        if result.returncode != 0 and "Could not create NMClient object" in result.stderr:
            self._logger.error("NetworkManager service is not accessible. Ensure D-Bus and NetworkManager services are running.")
            # Try to start services if possible
            try:
                subprocess.run(["service", "dbus", "start"], check=False, capture_output=True)
                subprocess.run(["service", "network-manager", "start"], check=False, capture_output=True)
                # Retry the original command once
                result = subprocess.run(cmd, check=False, text=True, capture_output=True)
            except Exception as e:
                self._logger.error(f"Failed to restart services: {e}")
        
        return result

    def _check_networkmanager_status(self) -> bool:
        """Check if NetworkManager is accessible and running."""
        try:
            result = subprocess.run([self._nmcli, "general", "status"], 
                                  check=False, text=True, capture_output=True)
            return result.returncode == 0
        except Exception:
            return False

    def _get_wifi_ifname(self) -> Optional[str]:
        # nmcli -t -f DEVICE,TYPE device status
        p = self._run(["-t", "-f", "DEVICE,TYPE", "device", "status"])
        if p.returncode != 0:
            self._logger.error(p.stderr.strip())
            return None
        for line in p.stdout.strip().splitlines():
            dev, typ = (line.split(":") + ["", ""])[:2]
            if typ == "wifi":
                return dev
        return None

    def _parse_scan(self, text: str) -> List[Dict]:
        # fields: SSID,SIGNAL,SECURITY,CHAN,FREQ
        nets: Dict[str, Dict] = {}
        for line in text.strip().splitlines():
            if not line:
                continue
            parts = line.split(":")
            ssid = parts[0]
            signal = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            security = parts[2] if len(parts) > 2 else ""
            chan = parts[3] if len(parts) > 3 else ""
            freq = parts[4] if len(parts) > 4 else ""
            entry = {
                "ssid": ssid if ssid else "(hidden)",
                "hidden": ssid == "",
                "signal": signal,
                "security": security,
                "channel": chan,
                "freq_mhz": freq,
            }
            key = entry["ssid"]
            if key not in nets or (entry["signal"] or -1) > (nets[key]["signal"] or -1):
                nets[key] = entry
        return sorted(nets.values(), key=lambda d: d["signal"] if d["signal"] is not None else -1, reverse=True)

    def _con_exists(self, name: str) -> bool:
        r = self._run(["-t", "-f", "NAME", "connection", "show"])
        if r.returncode != 0:
            return False
        return any(line.strip() == name for line in r.stdout.splitlines())

    def _con_active(self, name: str) -> bool:
        r = self._run(["-t", "-f", "NAME", "connection", "show", "--active"])
        if r.returncode != 0:
            return False
        return any(line.strip() == name for line in r.stdout.splitlines())

    def _get_if_ipv4(self, ifname: str) -> Optional[str]:
        r = self._run(["-t", "-f", "IP4.ADDRESS", "device", "show", ifname])
        if r.returncode != 0:
            return None
        # e.g. "IP4.ADDRESS[1]:10.42.0.1/24"
        for line in r.stdout.splitlines():
            if ":" in line:
                val = line.split(":", 1)[1].strip()
                if val and val != "--":
                    return val
        return None

    # ---------- API ----------

    @APIExport(runOnUIThread=False)
    def getNetworkManagerStatus(self) -> Dict:
        """Check if NetworkManager is running and accessible."""
        status = self._check_networkmanager_status()
        return {"running": status, "accessible": status}

    @APIExport(runOnUIThread=False)
    def scanNetworks(self, ifname: Optional[str] = None) -> Dict:
        # nmcli device wifi list
        ifname = ifname or self._get_wifi_ifname()
        args = ["-t", "-f", "SSID,SIGNAL,SECURITY,CHAN,FREQ", "device", "wifi", "list"]
        if ifname:
            args += ["ifname", ifname]
        p = self._run(args)
        if p.returncode != 0:
            err = p.stderr.strip() or "nmcli scan failed"
            self._logger.error(err)
            return {"error": err}
        self._last_scan = self._parse_scan(p.stdout)
        return {"networks": self._last_scan, "ifname": ifname}

    @APIExport(runOnUIThread=False)
    def getAvailableNetworks(self) -> List[Dict]:
        return self._last_scan

    @APIExport(runOnUIThread=False)
    def connectNetwork(self, ssid: str, password: Optional[str] = None, ifname: Optional[str] = None) -> Dict:
        if not ssid:
            return {"error": "SSID required"}
        ifname = ifname or self._get_wifi_ifname()

        # If an AP is active, bring it down before trying STA mode.
        if self._con_active("imswitch-hotspot"):
            self.stopAccessPoint()

        args = ["device", "wifi", "connect", ssid]
        if ifname:
            args += ["ifname", ifname]
        if password:
            args += ["password", password]

        p = self._run(args)
        if p.returncode != 0:
            msg = p.stderr.strip()
            self._logger.warning(f"Direct connect failed: {msg}. Trying to activate existing connection...")
            alt = self._run(["connection", "up", ssid] + (["ifname", ifname] if ifname else []))
            if alt.returncode != 0:
                return {"error": alt.stderr.strip() or msg}

        current = self.getCurrentSSID(ifname=ifname).get("ssid")
        if current == ssid:
            return {"status": "connected", "ssid": ssid, "ifname": ifname}
        return {"status": "requested", "ssid": ssid, "ifname": ifname}

    @APIExport(runOnUIThread=False)
    def getCurrentSSID(self, ifname: Optional[str] = None) -> Dict:
        ifname = ifname or self._get_wifi_ifname()
        if not ifname:
            return {"ssid": None}

        p = self._run(["-t", "-f", "GENERAL.CONNECTION", "device", "show", ifname])
        if p.returncode != 0:
            return {"ssid": None, "error": p.stderr.strip()}
        line = p.stdout.strip()
        ssid = None
        if ":" in line:
            conn_name = line.split(":", 1)[1].strip()
            if conn_name and conn_name != "--":
                ssid = conn_name
        return {"ssid": ssid, "ifname": ifname}

    # ---------- Access Point / Hotspot ----------

    @APIExport(runOnUIThread=False)
    def startAccessPoint(
        self,
        ssid: str,
        password: str,
        ifname: Optional[str] = None,
        con_name: str = "imswitch-hotspot",
        band: str = "bg",            # "bg" (2.4GHz) or "a" (5GHz), if supported
        channel: Optional[int] = None
    ) -> Dict:
        """
        Create & bring up an AP with given SSID/password.
        Returns: {"status":"up","ssid":..., "ifname":..., "con_name":..., "ipv4":...} or {"error": ...}
        """
        if not ssid:
            return {"error": "SSID required"}
        if not password or len(password) < 8:
            return {"error": "Password (WPA2/WPA3) must be ≥ 8 characters"}

        ifname = ifname or self._get_wifi_ifname()
        if not ifname:
            return {"error": "No Wi-Fi interface found"}

        # Disconnect from any current STA connection to free the radio.
        self._run(["device", "disconnect", ifname])

        # Clean any stale hotspot connection
        if self._con_active(con_name):
            self._run(["connection", "down", con_name])
        if self._con_exists(con_name):
            self._run(["connection", "delete", con_name])

        args = ["device", "wifi", "hotspot", "ifname", ifname, "con-name", con_name, "ssid", ssid, "password", password]
        # Optional band/channel (depends on driver/regdomain support)
        if band in ("a", "bg"):
            args += ["band", band]
        if channel is not None:
            args += ["channel", str(channel)]

        p = self._run(args)
        if p.returncode != 0:
            return {"error": p.stderr.strip() or "Failed to create hotspot"}

        # Ensure IPv4 sharing is enabled (usually done automatically by hotspot)
        # nmcli con modify <con_name> ipv4.method shared
        self._run(["connection", "modify", con_name, "ipv4.method", "shared"])

        # Bring it up (idempotent)
        up = self._run(["connection", "up", con_name])
        if up.returncode != 0:
            return {"error": up.stderr.strip() or "Failed to activate hotspot"}

        ipv4 = self._get_if_ipv4(ifname)  # typically 10.42.0.1/24
        return {"status": "up", "ssid": ssid, "ifname": ifname, "con_name": con_name, "ipv4": ipv4}

    @APIExport(runOnUIThread=False)
    def stopAccessPoint(self, con_name: str = "imswitch-hotspot") -> Dict:
        """
        Bring down and delete the AP connection.
        """
        if self._con_active(con_name):
            d = self._run(["connection", "down", con_name])
            if d.returncode != 0:
                return {"error": d.stderr.strip() or f"Failed to down {con_name}"}
        if self._con_exists(con_name):
            rm = self._run(["connection", "delete", con_name])
            if rm.returncode != 0:
                return {"error": rm.stderr.strip() or f"Failed to delete {con_name}"}
        return {"status": "removed", "con_name": con_name}
