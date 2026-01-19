#!/bin/bash
# Standalone Minimal Kiosk Setup für Raspberry Pi
# Kann direkt auf dem Pi ausgeführt werden - keine Projekt-Dependencies nötig

set -e

echo "=== ImSwitch Minimaler Kiosk-Modus Setup ==="
echo "Richtet automatischen Browser-Start im Vollbildmodus ein"
echo "URL: http://localhost:8001/imswitch/index.html?mode=kiosk"
echo ""

# Update system
echo "Step 1: System aktualisieren..."
sudo apt-get update

# Install minimal X server and window manager
echo "Step 2: X-Server und minimalen Window Manager installieren..."
sudo apt-get install -y \
    xserver-xorg \
    x11-xserver-utils \
    xinit \
    openbox \
    chromium-browser \
    unclutter

echo ""
echo "Step 3: Autostart-Konfiguration erstellen..."

# Create openbox autostart directory
mkdir -p ~/.config/openbox

# Create autostart script for Openbox
cat > ~/.config/openbox/autostart << 'EOF'
# Disable screen blanking and power management
xset s off
xset -dpms
xset s noblank

# Hide mouse cursor after 0.5s of inactivity
unclutter -idle 0.5 -root &

# Wait for ImSwitch server to be ready
sleep 5

# Start Chromium in kiosk mode with kiosk UI
chromium-browser \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-restore-session-state \
  --disable-translate \
  --kiosk \
  --incognito \
  --touch-events=enabled \
  http://localhost:8001/imswitch/index.html?mode=kiosk &
EOF

chmod +x ~/.config/openbox/autostart

echo ""
echo "Step 4: X-Server Autostart beim Boot einrichten..."

# Create .xinitrc to start openbox
cat > ~/.xinitrc << 'EOF'
#!/bin/sh
exec openbox-session
EOF

chmod +x ~/.xinitrc

# Add startx to .bash_profile for auto-login
if ! grep -q "startx" ~/.bash_profile 2>/dev/null; then
    cat >> ~/.bash_profile << 'EOF'

# Auto-start X server on login (only on tty1)
if [[ -z $DISPLAY ]] && [[ $(tty) = /dev/tty1 ]]; then
    exec startx
fi
EOF
fi

echo ""
echo "Step 5: Auto-Login auf tty1 einrichten..."

# Enable auto-login for user pi on tty1
sudo mkdir -p /etc/systemd/system/getty@tty1.service.d/
sudo tee /etc/systemd/system/getty@tty1.service.d/autologin.conf > /dev/null << EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $USER --noclear %I \$TERM
EOF

echo ""
echo "Step 6: Screen blanking deaktivieren..."

# Disable screen blanking in kernel boot options
if ! grep -q "consoleblank=0" /boot/cmdline.txt 2>/dev/null && ! grep -q "consoleblank=0" /boot/firmware/cmdline.txt 2>/dev/null; then
    if [ -f /boot/cmdline.txt ]; then
        sudo sed -i '$ s/$/ consoleblank=0/' /boot/cmdline.txt
    elif [ -f /boot/firmware/cmdline.txt ]; then
        sudo sed -i '$ s/$/ consoleblank=0/' /boot/firmware/cmdline.txt
    fi
fi

echo ""
echo "=== Setup abgeschlossen! ==="
echo ""
echo "Nach dem Neustart wird:"
echo "  1. Automatisch eingeloggt (User: $USER)"
echo "  2. X-Server mit Openbox gestartet"
echo "  3. Chromium im Kiosk-Modus geöffnet"
echo "  4. ImSwitch-WebApp im Kiosk-UI geladen (800x480 optimiert)"
echo ""
echo "WICHTIG: Stelle sicher, dass der ImSwitch-Server automatisch startet!"
echo ""
echo "Neustart mit: sudo reboot"
echo ""
echo "=== Kiosk-Modus beenden ==="
echo "  Ctrl+Alt+F2  - Zu Terminal wechseln"
echo "  Ctrl+Alt+F1  - Zurück zum Kiosk"
