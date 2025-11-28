# Docker Configuration Guide

## ✅ Changes Made

### 1. **CONFIG_FILE is now OPTIONAL**
- If not provided, ImSwitch will use the default setup from `imcontrol_options.json`
- This file is located at `CONFIG_PATH/config/imcontrol_options.json`
- The default setup is automatically selected from available configs in `CONFIG_PATH/imcontrol_setups/`

### 2. **Path Validation Fixed**
- `entrypoint.sh` no longer fails prematurely when volumes don't exist yet
- Docker volumes will be created automatically if they don't exist
- Warnings are shown instead of hard errors for missing directories

### 3. **Build Optimization**
- Added `.dockerignore` to exclude unnecessary files from build context
- Created `docker/build-optimized.sh` script for faster incremental builds
- Builds now cache layers properly using BuildKit

---

## 📋 Environment Variables Reference

### Required Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `CONFIG_PATH` | Path to ImSwitch config directory | `/home/pi/ImSwitchConfig` |
| `DATA_PATH` | Default data storage path | `/home/pi/Datasets` |

### Optional Variables
| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CONFIG_FILE` | Specific config file to use | Uses `imcontrol_options.json` | `example_virtual_microscope.json` |
| `HEADLESS` | Run without GUI | `false` | `1`, `true` |
| `HTTP_PORT` | HTTP server port | `8001` | `8080` |
| `SSL` | Enable SSL | `true` | `0`, `false` |
| `SCAN_EXT_DATA_PATH` | Scan for external drives | `false` | `1`, `true` |
| `EXT_DATA_PATH` | External mount point directory | - | `/media`, `/Volumes` |

---

## 🚀 Usage Examples

### Option 1: Use Default Config (Recommended)
```yaml
# docker-compose.yml
environment:
  - CONFIG_PATH=/home/pi/ImSwitchConfig
  - DATA_PATH=/home/pi/Datasets
  # No CONFIG_FILE needed - uses imcontrol_options.json
```

### Option 2: Specify Config File
```yaml
# docker-compose.yml
environment:
  - CONFIG_PATH=/home/pi/ImSwitchConfig
  - CONFIG_FILE=example_virtual_microscope.json
  - DATA_PATH=/home/pi/Datasets
```

### Option 3: Use Absolute Path to Config
```yaml
# docker-compose.yml
environment:
  - CONFIG_PATH=/home/pi/ImSwitchConfig
  - CONFIG_FILE=/home/pi/ImSwitchConfig/imcontrol_setups/my_custom_setup.json
  - DATA_PATH=/home/pi/Datasets
```

---

## 🏗️ Building the Docker Image

### Fast Incremental Build (Recommended)
```bash
cd /path/to/ImSwitch
./docker/build-optimized.sh
```

### Force Complete Rebuild
```bash
./docker/build-optimized.sh --force
```

### Build Without Any Cache
```bash
./docker/build-optimized.sh --no-cache
```

### Manual Build (Old Way)
```bash
# Still works, but slower
DOCKER_BUILDKIT=1 docker build -t imswitch-holo .
```

---

## 🔄 Configuration Loading Flow

```
1. Docker Compose Sets Environment Variables
   ├─ CONFIG_PATH (required)
   ├─ CONFIG_FILE (optional)
   └─ DATA_PATH (required)
   
2. entrypoint.sh Validates and Builds CLI Args
   ├─ Checks CONFIG_PATH exists
   ├─ Warns if CONFIG_FILE doesn't exist (non-fatal)
   └─ Passes args to Python CLI
   
3. Python CLI Parser (imswitch/__main__.py)
   ├─ Parses --config-folder, --config-file, --data-folder
   └─ Updates global config object
   
4. configfiletools.py Loads Configuration
   ├─ If DEFAULT_SETUP_FILE is set → use it
   ├─ Else if imcontrol_options.json exists → use saved default
   └─ Else → use first available .json in imcontrol_setups/
```

---

## 🐛 Troubleshooting

### "Error: Configuration path not provided"
**Solution:** Set `CONFIG_PATH` in docker-compose.yml
```yaml
environment:
  - CONFIG_PATH=/home/pi/ImSwitchConfig
```

### "Warning: Configuration file does not exist"
**Solution:** Either:
1. Don't set `CONFIG_FILE` (use default from `imcontrol_options.json`)
2. Check the file exists in `CONFIG_PATH/imcontrol_setups/`
3. Use absolute path: `CONFIG_FILE=/full/path/to/file.json`

### Build Takes Too Long
**Solution:** Use the optimized build script:
```bash
./docker/build-optimized.sh
```

### Need to Force Rebuild
**Solution:** 
```bash
./docker/build-optimized.sh --force
```

---

## 📂 Directory Structure

```
/home/pi/ImSwitchConfig/          # CONFIG_PATH
├── config/
│   └── imcontrol_options.json    # Default config selection
└── imcontrol_setups/             # Available setup files
    ├── example_virtual_microscope.json
    ├── my_custom_setup.json
    └── ...

/home/pi/Datasets/                # DATA_PATH
└── [recorded data]

/media/                           # EXT_DATA_PATH (optional)
└── [external USB drives]
```

---

## ⚙️ Build Optimization Details

The new build process uses Docker BuildKit with layer caching and UV package manager:

1. **Base layers** (OS, Python 3.11, UV, drivers) - Rarely change, cached
2. **Dependencies** (Python packages via UV) - Change occasionally, cached unless pyproject.toml changes
3. **Application code** (ImSwitch) - Changes frequently, rebuilt only when needed

**Speed improvements:**
- First build: ~15-30 minutes (no cache)
- Incremental build: ~2-5 minutes (with cache)
- No-op rebuild: ~10 seconds (if nothing changed)

The `BUILD_DATE` argument is only added with `--force` flag, preventing unnecessary cache invalidation.
