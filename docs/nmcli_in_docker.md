Use this minimal pair (no `--separator`):

**Dockerfile**

```dockerfile
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 dbus network-manager ca-certificates iw iproute2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY scan.py .

CMD ["python3", "scan.py"]
```

**scan.py**

```python
#!/usr/bin/env python3
import json, shutil, subprocess, sys

FIELDS = "SSID,SIGNAL,SECURITY,CHAN,FREQ,BSSID"

def run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True)

def split_escaped(s: str, sep=":"):
    out, cur, esc = [], [], False
    for ch in s:
        if esc:
            cur.append(ch); esc = False
        elif ch == "\\":
            esc = True
        elif ch == sep:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    # unescape nmcli-style backslashes
    out = [x.replace("\\"+sep, sep).replace("\\\\", "\\") for x in out]
    return out

def nmcli_scan():
    nmcli = shutil.which("nmcli")
    if not nmcli:
        return None, "nmcli not found"

    # Try with escaping (older nmcli may not accept --escape)
    attempts = [
        [nmcli, "-t", "--escape", "yes", "-f", FIELDS, "device", "wifi", "list"],
        [nmcli, "-t", "-f", FIELDS, "device", "wifi", "list"],
    ]
    last_err, out = None, None
    for cmd in attempts:
        p = run(cmd)
        if p.returncode == 0:
            out = p.stdout
            break
        last_err = (p.stderr or p.stdout or "").strip()

    if out is None:
        return None, last_err or "nmcli failed"

    nets = []
    for line in out.splitlines():
        if not line: continue
        parts = split_escaped(line)
        parts += [""] * (6 - len(parts))
        ssid, signal, security, chan, freq, bssid = parts[:6]
        try:
            signal = int(signal) if signal else None
        except ValueError:
            signal = None
        nets.append({
            "ssid": ssid or "(hidden)",
            "signal": signal,
            "security": security,
            "channel": chan,
            "freq_mhz": freq,
            "bssid": bssid,
        })
    return nets, None

def main():
    nets, err = nmcli_scan()
    if err:
        print(json.dumps({
            "error": err,
            "hint": "Run container with --net=host and mount /run/dbus/system_bus_socket and /etc/machine-id. "
                    "Ensure NetworkManager is active on the host and a Wi-Fi device exists."
        }, indent=2)); sys.exit(1)
    print(json.dumps({"networks": nets}, indent=2))
    sys.exit(0 if nets else 2)

if __name__ == "__main__":
    main()
```

**Build**

```bash
docker build -t wifi-scan .
```

**Run (talk to host NetworkManager via D-Bus)**

```bash
docker run --rm -it \
  --net=host \
  -v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket \
  -v /etc/machine-id:/etc/machine-id:ro \
  wifi-scan
```

**Host prerequisites (outside container)**

```bash
systemctl is-active NetworkManager           # should be "active"
nmcli -g DEVICE,TYPE,STATE dev               # should show a wifi device (wlan0/wlp*)
rfkill list                                  # wifi not soft/hard blocked
```

**If you only need a raw scan (no NM)**

```bash
docker run --rm -it --net=host --privileged wifi-scan bash -lc 'iw dev; iw dev wlan0 scan | head'
```
