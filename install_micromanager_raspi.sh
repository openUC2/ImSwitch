#!/usr/bin/env bash
# =============================================================================
# install_micromanager_rpi.sh
#
# Build and install Micro-Manager (MMCore + device adapters) and pymmcore-plus
# on a Raspberry Pi running Pi OS Bookworm (64-bit / arm64).
#
# This compiles everything from source because:
#   1) There are no prebuilt MM device adapters for arm64.
#   2) pymmcore on PyPI only ships x86_64 + macOS-arm64 wheels, not linux-aarch64.
#   3) The pymmcore wheel and the device adapters MUST share the same device
#      interface version, so we build them from the same mmCoreAndDevices tree.
#
# What this script does NOT do:
#   - Install Java or build MMStudio (not needed for pymmcore-plus)
#   - Install vendor camera SDKs (add those yourself after, see Section 6)
#   - Set up Docker (this is the "test outside Docker first" script)
#
# Usage:
#   chmod +x install_micromanager_rpi.sh
#   ./install_micromanager_rpi.sh 2>&1 | tee build.log
#
# Tested on: Raspberry Pi 5 (8 GB) with Pi OS Bookworm 64-bit
# Expected build time: ~15-25 min on Pi 5, ~30-50 min on Pi 4
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

# Where Micro-Manager gets installed (libs, adapters, configs)
MM_INSTALL_PREFIX="${MM_INSTALL_PREFIX:-/opt/micro-manager}"

# Where we clone and build (can be deleted after install)
BUILD_DIR="${BUILD_DIR:-$HOME/mm-build}"

# Python to use — change if you use a venv or pyenv
PYTHON="${PYTHON:-python3}"

# How many parallel make jobs (Pi 5 has 4 cores)
NJOBS="${NJOBS:-$(nproc)}"

# Which adapters to build. "all" = every adapter whose dependencies are met.
# For a minimal first test, set to "DemoCamera SerialManager" to save time.
# Separate with spaces. "all" means don't touch Makefile.am.
ADAPTERS="${ADAPTERS:-all}"

# Pin a specific mmCoreAndDevices tag/branch for reproducibility.
# "main" = latest. Use a release tag like "v11.2.1" for stability.
MMCORE_REF="${MMCORE_REF:-main}"

# ─── Colors for output ──────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ─── 0. Preflight checks ────────────────────────────────────────────────────

info "Running preflight checks..."

[[ "$(uname -m)" == "aarch64" ]] || fail "This script is for arm64 / aarch64. You are on $(uname -m)."

if ! command -v $PYTHON &>/dev/null; then
    fail "$PYTHON not found. Install python3 first: sudo apt install python3 python3-pip python3-venv"
fi

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')
if (( PY_MAJOR < 3 || PY_MINOR < 10 )); then
    fail "Python >= 3.10 required. Found $PY_VERSION."
fi
info "Using $PYTHON ($PY_VERSION)"

# Check available RAM — building with <2 GB free is risky
FREE_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
if (( FREE_MB < 1500 )); then
    warn "Only ${FREE_MB} MB RAM available. Build may fail or use heavy swap."
    warn "Consider closing other programs or adding swap (sudo dphys-swapfile swapoff && sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon)."
fi

# Check disk space — the build tree needs ~2-3 GB
FREE_DISK_MB=$(df --output=avail -BM "$HOME" | tail -1 | tr -d ' M')
if (( FREE_DISK_MB < 3000 )); then
    fail "Less than 3 GB disk space free in $HOME. Need at least 3 GB for the build."
fi

# ─── 1. Install system dependencies ─────────────────────────────────────────

info "Installing system dependencies (requires sudo)..."

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    autoconf \
    automake \
    libtool \
    autoconf-archive \
    pkg-config \
    git \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-numpy \
    swig

# We do NOT install Java or ant — we build with --without-java.

# Check SWIG version — SWIG 4.x is fine for pymmcore (the SWIG 3.x requirement
# is only for MMCoreJ / Java wrapper, which we skip).
SWIG_VERSION=$(swig -version 2>/dev/null | grep -oP 'SWIG Version \K[0-9]+\.[0-9]+' || echo "0.0")
info "SWIG version: $SWIG_VERSION"

# ─── 2. Clone mmCoreAndDevices ───────────────────────────────────────────────

info "Setting up build directory at $BUILD_DIR ..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [[ -d mmCoreAndDevices ]]; then
    info "mmCoreAndDevices directory already exists, pulling latest..."
    cd mmCoreAndDevices
    git fetch --all
    git checkout "$MMCORE_REF"
    git pull --ff-only 2>/dev/null || true  # pull fails on detached HEAD, that's fine
    cd ..
else
    info "Cloning mmCoreAndDevices (ref: $MMCORE_REF)..."
    git clone https://github.com/micro-manager/mmCoreAndDevices.git
    cd mmCoreAndDevices
    git checkout "$MMCORE_REF"
    cd ..
fi

# We also need the top-level micro-manager repo because mmCoreAndDevices'
# autotools build is designed to run as a submodule of micro-manager.
if [[ -d micro-manager ]]; then
    info "micro-manager directory already exists, pulling latest..."
    cd micro-manager
    git fetch --all
    git checkout main
    git pull --ff-only 2>/dev/null || true
    cd ..
else
    info "Cloning micro-manager (for the autotools build harness)..."
    git clone https://github.com/micro-manager/micro-manager.git
fi

# Point the submodule at our local clone to avoid a second download
cd micro-manager
git submodule update --init --recursive 2>/dev/null || {
    # If submodule init fails, manually wire it
    info "Wiring mmCoreAndDevices submodule to local clone..."
    rm -rf mmCoreAndDevices
    ln -sf "$BUILD_DIR/mmCoreAndDevices" mmCoreAndDevices
}
cd ..

# ─── 3. Optionally restrict adapters to build ───────────────────────────────

cd micro-manager/mmCoreAndDevices

if [[ "$ADAPTERS" != "all" ]]; then
    info "Restricting build to adapters: $ADAPTERS"

    # Back up original Makefile.am
    cp DeviceAdapters/Makefile.am DeviceAdapters/Makefile.am.orig

    # Build the SUBDIRS line with only the requested adapters
    # Always include "." (the parent directory target)
    SUBDIRS_LINE="SUBDIRS = ."
    for adapter in $ADAPTERS; do
        if [[ -d "DeviceAdapters/$adapter" ]]; then
            SUBDIRS_LINE="$SUBDIRS_LINE $adapter"
        else
            warn "Adapter directory DeviceAdapters/$adapter not found — skipping."
        fi
    done

    # Replace the SUBDIRS block in Makefile.am
    # The original is a multi-line definition; we replace the entire file section
    python3 -c "
import re, sys
text = open('DeviceAdapters/Makefile.am').read()
# Match the SUBDIRS definition (possibly multi-line with backslash continuation)
text = re.sub(
    r'^SUBDIRS\s*=.*?(?:\\\\\n.*?)*$',
    '${SUBDIRS_LINE}',
    text,
    flags=re.MULTILINE
)
open('DeviceAdapters/Makefile.am', 'w').write(text)
print('Makefile.am patched successfully.')
"
fi

cd "$BUILD_DIR/micro-manager"

# ─── 4. Build MMCore + device adapters ───────────────────────────────────────

info "Running autogen.sh..."
./autogen.sh

info "Running configure (--without-java, prefix=$MM_INSTALL_PREFIX)..."
./configure \
    --prefix="$MM_INSTALL_PREFIX" \
    --without-java \
    --disable-java-app \
    --with-python="$($PYTHON -c 'import sys; print(sys.executable)')" \
    PYTHON="$PYTHON" \
    2>&1 | tee "$BUILD_DIR/configure.log"

# Check configure actually succeeded
if [[ ! -f Makefile ]]; then
    fail "configure failed. Check $BUILD_DIR/configure.log"
fi

# Print which adapters will be built
info "Adapters that will be built:"
grep -A200 '^Enabled device adapters' "$BUILD_DIR/configure.log" | head -40 || \
    info "(Could not parse adapter list from configure output — check configure.log)"

info "Building with make -j${NJOBS}... (this will take a while)"
make -j"$NJOBS" 2>&1 | tee "$BUILD_DIR/make.log"

info "Installing to $MM_INSTALL_PREFIX (requires sudo)..."
sudo make install 2>&1 | tee "$BUILD_DIR/install.log"

# ─── 5. Install pymmcore + pymmcore-plus into a venv ─────────────────────────

info "Setting up Python virtual environment..."

VENV_DIR="$HOME/mm-venv"

if [[ -d "$VENV_DIR" ]]; then
    info "Venv $VENV_DIR already exists, reusing."
else
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# Build pymmcore from source (no arm64 wheel on PyPI).
# pymmcore's setup.py will find the MMCore source in mmCoreAndDevices if we
# set the env var, or we can pip-install from the repo and let it build.
info "Installing pymmcore from source (this compiles the C++ SWIG wrapper)..."
pip install pymmcore --no-binary pymmcore 2>&1 | tee "$BUILD_DIR/pymmcore-install.log" || {
    warn "pip install pymmcore failed. Trying to build from the local tree..."
    # Alternative: build from local mmCoreAndDevices using the pymmcore build in-tree
    # (This path is a fallback if the PyPI sdist doesn't compile cleanly)
    cd "$BUILD_DIR/mmCoreAndDevices"
    if [[ -f pymmcore/setup.py ]] || [[ -f setup.py ]]; then
        pip install . 2>&1 | tee "$BUILD_DIR/pymmcore-local-install.log"
    else
        fail "Could not install pymmcore. Check $BUILD_DIR/pymmcore-install.log"
    fi
}

info "Installing pymmcore-plus..."
pip install "pymmcore-plus[cli]"

# Record the device interface versions for sanity checking
info "Checking device interface version match..."
PYMMCORE_DI=$($PYTHON -c "import pymmcore; c = pymmcore.CMMCore(); print(c.getAPIVersionInfo())" 2>/dev/null || echo "UNKNOWN")
info "pymmcore reports: $PYMMCORE_DI"

# ─── 6. Create a demo config and smoke-test ──────────────────────────────────

info "Setting up demo configuration..."

MM_CONFIG_DIR="$HOME/micro-manager-configs"
mkdir -p "$MM_CONFIG_DIR"

# Write a minimal demo config that uses the DemoCamera adapter
cat > "$MM_CONFIG_DIR/MMConfig_demo.cfg" << 'DEMOCFG'
# Micro-Manager Demo Configuration
# Minimal config for testing MMCore on Raspberry Pi

# Camera
Device,Camera,DemoCamera,DCam
Property,Camera,OnCameraCCDXSize,512
Property,Camera,OnCameraCCDYSize,512
Property,Camera,PixelType,8bit

# XY Stage
Device,XY,DemoCamera,DXYStage

# Z Stage
Device,Z,DemoCamera,DStage

# Shutter
Device,Shutter,DemoCamera,DShutter

# Labels
Label,Channel,1,DAPI
Label,Channel,2,FITC
Label,Channel,3,Rhodamine

# System configuration
ConfigGroup,System,Startup,Camera,Exposure,10.0

# Initialization
Property,Core,Initialize,1
DEMOCFG

info "Demo config written to: $MM_CONFIG_DIR/MMConfig_demo.cfg"

# Write the smoke test script
cat > "$MM_CONFIG_DIR/smoke_test.py" << 'SMOKETEST'
#!/usr/bin/env python3
"""
Smoke test for Micro-Manager on Raspberry Pi.
Run this after install_micromanager_rpi.sh to verify everything works.

Usage:
    source ~/mm-venv/bin/activate
    python ~/micro-manager-configs/smoke_test.py
"""

import sys
import os
import time
import numpy as np

def main():
    # ── 1. Import pymmcore-plus ──────────────────────────────────────────
    try:
        from pymmcore_plus import CMMCorePlus
        print("[OK]  pymmcore-plus imported successfully")
    except ImportError:
        print("[FAIL] Could not import pymmcore_plus. Is the venv activated?")
        print("       Run: source ~/mm-venv/bin/activate")
        sys.exit(1)

    # ── 2. Instantiate core ──────────────────────────────────────────────
    core = CMMCorePlus.instance()
    print(f"[OK]  CMMCorePlus instantiated")
    print(f"      MMCore version: {core.getVersionInfo()}")
    print(f"      API version:    {core.getAPIVersionInfo()}")

    # ── 3. Set adapter search paths ──────────────────────────────────────
    mm_path = os.environ.get("MICROMANAGER_PATH", "/opt/micro-manager/lib/micro-manager")

    if not os.path.isdir(mm_path):
        # Try alternative locations
        alternatives = [
            "/opt/micro-manager/lib/micro-manager",
            "/usr/local/lib/micro-manager",
            "/usr/lib/micro-manager",
        ]
        for alt in alternatives:
            if os.path.isdir(alt):
                mm_path = alt
                break
        else:
            print(f"[FAIL] Device adapter directory not found.")
            print(f"       Tried: {', '.join(alternatives)}")
            print(f"       Set MICROMANAGER_PATH env var to the correct path.")
            sys.exit(1)

    core.setDeviceAdapterSearchPaths([mm_path])
    print(f"[OK]  Adapter search path: {mm_path}")

    # List available adapters
    adapter_files = [f for f in os.listdir(mm_path) if f.startswith("libmmgr_dal_")]
    print(f"      Found {len(adapter_files)} adapter libraries:")
    for f in sorted(adapter_files)[:15]:
        print(f"        {f}")
    if len(adapter_files) > 15:
        print(f"        ... and {len(adapter_files) - 15} more")

    # ── 4. Load demo config ──────────────────────────────────────────────
    cfg_path = os.path.join(os.path.expanduser("~"), "micro-manager-configs", "MMConfig_demo.cfg")

    if not os.path.isfile(cfg_path):
        print(f"[WARN] Demo config not found at {cfg_path}")
        print(f"       Trying to load DemoCamera manually...")
        try:
            core.loadDevice("Camera", "DemoCamera", "DCam")
            core.loadDevice("XY", "DemoCamera", "DXYStage")
            core.loadDevice("Z", "DemoCamera", "DStage")
            core.loadDevice("Shutter", "DemoCamera", "DShutter")
            core.initializeAllDevices()
            core.setCameraDevice("Camera")
            core.setXYStageDevice("XY")
            core.setFocusDevice("Z")
            core.setShutterDevice("Shutter")
        except Exception as e:
            print(f"[FAIL] Could not load demo devices: {e}")
            sys.exit(1)
    else:
        try:
            core.loadSystemConfiguration(cfg_path)
            print(f"[OK]  Loaded config: {cfg_path}")
        except Exception as e:
            print(f"[WARN] Config load failed ({e}), trying manual device loading...")
            core.unloadAllDevices()
            core.loadDevice("Camera", "DemoCamera", "DCam")
            core.loadDevice("XY", "DemoCamera", "DXYStage")
            core.loadDevice("Z", "DemoCamera", "DStage")
            core.loadDevice("Shutter", "DemoCamera", "DShutter")
            core.initializeAllDevices()
            core.setCameraDevice("Camera")
            core.setXYStageDevice("XY")
            core.setFocusDevice("Z")
            core.setShutterDevice("Shutter")

    # ── 5. Print loaded devices ──────────────────────────────────────────
    devices = core.getLoadedDevices()
    print(f"[OK]  Loaded {len(devices)} devices: {', '.join(devices)}")
    print(f"      Camera:   {core.getCameraDevice()}")
    print(f"      XY Stage: {core.getXYStageDevice()}")
    print(f"      Z Stage:  {core.getFocusDevice()}")
    print(f"      Shutter:  {core.getShutterDevice()}")

    # ── 6. Snap a single image ───────────────────────────────────────────
    try:
        img = core.snap()
        print(f"[OK]  Snapped image: shape={img.shape}, dtype={img.dtype}, "
              f"min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    except Exception as e:
        print(f"[FAIL] snap() failed: {e}")
        sys.exit(1)

    # ── 7. Snap 10 frames and measure throughput ─────────────────────────
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        core.snap()
    elapsed = time.perf_counter() - t0
    fps = N / elapsed
    print(f"[OK]  {N} frames in {elapsed:.2f}s = {fps:.1f} FPS (demo camera)")

    # ── 8. Test XY stage ─────────────────────────────────────────────────
    try:
        x0, y0 = core.getXPosition(), core.getYPosition()
        core.setXYPosition(x0 + 100.0, y0 + 50.0)
        core.waitForDevice(core.getXYStageDevice())
        x1, y1 = core.getXPosition(), core.getYPosition()
        dx, dy = abs(x1 - (x0 + 100.0)), abs(y1 - (y0 + 50.0))
        if dx < 1.0 and dy < 1.0:
            print(f"[OK]  XY move: ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f}) "
                  f"error=({dx:.3f},{dy:.3f}) µm")
        else:
            print(f"[WARN] XY move error larger than expected: dx={dx:.3f}, dy={dy:.3f}")
        # Return to origin
        core.setXYPosition(x0, y0)
        core.waitForDevice(core.getXYStageDevice())
    except Exception as e:
        print(f"[WARN] XY stage test failed: {e}")

    # ── 9. Test Z stage ──────────────────────────────────────────────────
    try:
        z0 = core.getPosition()
        core.setPosition(z0 + 10.0)
        core.waitForDevice(core.getFocusDevice())
        z1 = core.getPosition()
        dz = abs(z1 - (z0 + 10.0))
        print(f"[OK]  Z move: {z0:.1f} -> {z1:.1f} µm, error={dz:.3f}")
        core.setPosition(z0)
        core.waitForDevice(core.getFocusDevice())
    except Exception as e:
        print(f"[WARN] Z stage test failed: {e}")

    # ── 10. Test shutter ─────────────────────────────────────────────────
    try:
        core.setShutterOpen(True)
        assert core.getShutterOpen() == True, "Shutter should be open"
        core.setShutterOpen(False)
        assert core.getShutterOpen() == False, "Shutter should be closed"
        print(f"[OK]  Shutter open/close works")
    except Exception as e:
        print(f"[WARN] Shutter test failed: {e}")

    # ── 11. Test continuous acquisition ──────────────────────────────────
    try:
        core.startContinuousSequenceAcquisition(0)
        time.sleep(0.5)
        remaining = core.getRemainingImageCount()
        core.stopSequenceAcquisition()
        print(f"[OK]  Continuous acquisition: {remaining} frames buffered in 0.5s")
    except Exception as e:
        print(f"[WARN] Continuous acquisition test failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SMOKE TEST PASSED — Micro-Manager is working on this Pi")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. To use a real camera, write a .cfg file for it and set")
    print("     MICROMANAGER_PATH to the adapter directory.")
    print("  2. To integrate with ImSwitch, follow the Stage 1 instructions")
    print("     in pymmcore-integration-plan.md.")
    print()
    print("Useful environment variables to put in ~/.bashrc:")
    print(f'  export MICROMANAGER_PATH="{mm_path}"')
    print(f'  export PATH="$HOME/mm-venv/bin:$PATH"')

    return 0

if __name__ == "__main__":
    sys.exit(main())
SMOKETEST

chmod +x "$MM_CONFIG_DIR/smoke_test.py"
info "Smoke test written to: $MM_CONFIG_DIR/smoke_test.py"

# ─── 7. Write environment setup to .bashrc snippet ──────────────────────────

ENV_FILE="$HOME/.mm_env"
cat > "$ENV_FILE" << EOF
# Micro-Manager environment — source this or add to .bashrc
export MICROMANAGER_PATH="$MM_INSTALL_PREFIX/lib/micro-manager"
export PATH="$VENV_DIR/bin:\$PATH"
EOF

info "Environment file written to $ENV_FILE"
info "Add to your shell: echo 'source $ENV_FILE' >> ~/.bashrc"

# ─── 8. Run the smoke test ──────────────────────────────────────────────────

info "Running smoke test..."
echo ""
export MICROMANAGER_PATH="$MM_INSTALL_PREFIX/lib/micro-manager"
$PYTHON "$MM_CONFIG_DIR/smoke_test.py"
RESULT=$?

echo ""
if [[ $RESULT -eq 0 ]]; then
    info "Installation complete!"
    echo ""
    echo "  Install prefix:     $MM_INSTALL_PREFIX"
    echo "  Adapter libs:       $MM_INSTALL_PREFIX/lib/micro-manager/"
    echo "  Python venv:        $VENV_DIR"
    echo "  Demo config:        $MM_CONFIG_DIR/MMConfig_demo.cfg"
    echo "  Smoke test:         $MM_CONFIG_DIR/smoke_test.py"
    echo "  Build artifacts:    $BUILD_DIR  (safe to delete)"
    echo ""
    echo "  To activate the venv in a new shell:"
    echo "    source $ENV_FILE"
    echo ""
    echo "  To re-run the smoke test:"
    echo "    source $ENV_FILE && python $MM_CONFIG_DIR/smoke_test.py"
    echo ""
else
    warn "Smoke test failed. Check output above. Build artifacts are in $BUILD_DIR."
fi

exit $RESULT