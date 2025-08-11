import subprocess
import shutil
from typing import List, Dict, Optional
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController


class WiFiController(ImConWidgetController):
    """
    Linux/Raspberry Pi only (uses `sudo nmcli`).
    Exposes:
      - scanNetworks() -> list of networks
      - getAvailableNetworks() -> last scan result
      - connectNetwork(ssid, password=None, ifname=None) -> status
      - getCurrentSSID(ifname=None) -> ssid or None
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._last_scan: List[Dict] = []
        self._sudo = shutil.which("sudo") or "sudo"
        self._nmcli = shutil.which("nmcli") or "nmcli"

    # ---------- helpers ----------

    def _run(self, args: List[str]) -> subprocess.CompletedProcess:
        cmd = [self._sudo, self._nmcli] + args
        self._logger.debug(f"Running: {' '.join(cmd)}")
        return subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )

    def _get_wifi_ifname(self) -> Optional[str]:
        # nmcli -t -f DEVICE,TYPE,STATE device status
        p = self._run(["-t", "-f", "DEVICE,TYPE,STATE", "device", "status"])
        if p.returncode != 0:
            self._logger.error(p.stderr.strip())
            return None
        for line in p.stdout.strip().splitlines():
            # e.g. "wlan0:wifi:connected"
            parts = line.split(":")
            if len(parts) >= 3 and parts[1] == "wifi":
                return parts[0]
        return None

    def _parse_scan(self, text: str) -> List[Dict]:
        # fields: SSID,SIGNAL,SECURITY,CHAN,FREQ
        nets: Dict[str, Dict] = {}
        for line in text.strip().splitlines():
            if not line:
                continue
            parts = line.split(":")
            # SSID can be empty; keep but mark as hidden
            ssid = parts[0]
            signal = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            security = parts[2] if len(parts) > 2 else ""
            chan = parts[3] if len(parts) > 3 else ""
            freq = parts[4] if len(parts) > 4 else ""
            entry = {
                "ssid": ssid if ssid else "(hidden)",
                "hidden": ssid == "",
                "signal": signal,
                "security": security,  # e.g. WPA2, WPA3, WEP, --
                "channel": chan,
                "freq_mhz": freq,
            }
            # Deduplicate by SSID, keep strongest
            key = entry["ssid"]
            if key not in nets or (entry["signal"] or -1) > (nets[key]["signal"] or -1):
                nets[key] = entry
        return sorted(nets.values(), key=lambda d: d["signal"] if d["signal"] is not None else -1, reverse=True)

    # ---------- API ----------

    @APIExport(runOnUIThread=False)
    def scanNetworks(self, ifname: Optional[str] = None) -> Dict:
        """
        Returns: {"networks":[{ssid, hidden, signal, security, channel, freq_mhz}], "ifname": "..."}
        """
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
        """Return last scan result without rescanning."""
        return self._last_scan

    @APIExport(runOnUIThread=False)
    def connectNetwork(self, ssid: str, password: Optional[str] = None, ifname: Optional[str] = None) -> Dict:
        """
        Connect by SSID. For open networks leave password=None.
        Returns: {"status":"connected","ssid":..., "ifname":...} or {"error": "..."}
        """
        if not ssid:
            return {"error": "SSID required"}

        ifname = ifname or self._get_wifi_ifname()
        # Prefer interface-bound connect if available
        args = ["device", "wifi", "connect", ssid]
        if ifname:
            args += ["ifname", ifname]
        if password:
            args += ["password", password]

        p = self._run(args)
        if p.returncode != 0:
            # If connection name exists, try activating it
            msg = p.stderr.strip()
            self._logger.warning(f"Direct connect failed: {msg}. Trying to activate existing connection...")
            alt = self._run(["connection", "up", ssid] + (["ifname", ifname] if ifname else []))
            if alt.returncode != 0:
                return {"error": alt.stderr.strip() or msg}
        # success -> verify
        current = self.getCurrentSSID(ifname=ifname).get("ssid")
        if current == ssid:
            return {"status": "connected", "ssid": ssid, "ifname": ifname}
        return {"status": "requested", "ssid": ssid, "ifname": ifname}

    @APIExport(runOnUIThread=False)
    def getCurrentSSID(self, ifname: Optional[str] = None) -> Dict:
        """
        Returns: {"ssid": "...", "ifname": "..."} or {"ssid": None}
        """
        ifname = ifname or self._get_wifi_ifname()
        if not ifname:
            return {"ssid": None}

        # nmcli -t -f GENERAL.CONNECTION device show wlan0
        p = self._run(["-t", "-f", "GENERAL.CONNECTION", "device", "show", ifname])
        if p.returncode != 0:
            return {"ssid": None, "error": p.stderr.strip()}
        # Output like: "GENERAL.CONNECTION:My-WiFi"
        line = p.stdout.strip()
        ssid = None
        if ":" in line:
            conn_name = line.split(":", 1)[1].strip()
            if conn_name and conn_name != "--":
                ssid = conn_name
        return {"ssid": ssid, "ifname": ifname}
