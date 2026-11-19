# UV Usage Guide for ImSwitch

This document provides guidance on using UV with ImSwitch for faster package management and reproducible builds.

## Installation

Install UV following the official instructions:

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Development Workflow

### Setting up a development environment

```bash
# Clone the repository
git clone https://github.com/openUC2/ImSwitch.git
cd ImSwitch

# Create a virtual environment (on a Raspberry Pi: --system-site-packages, for picamera2)
uv venv

# Install ImSwitch with every dependency at the version in uv.lock (dev: pytest)
uv sync --extra dev
```

### Lock Files for Reproducible Builds

ImSwitch includes a `uv.lock` file for reproducible builds. To use it:

```bash
# Install from lock file (ensures exact versions)
uv sync

# Update lock file after changing dependencies
uv lock

# Move one dependency, e.g. UC2-REST, to its newest release
uv lock --upgrade-package uc2-rest
```

Do not fix up the environment with `uv pip install` afterwards: the next `uv sync` or `uv run`
undoes it. For the same reason, an editable UC2-REST (`uv pip install -e ../UC2-REST`) needs
`uv run --no-sync`.

**Note**: When contributing code, commit the `uv.lock` file to ensure all developers use identical dependency versions.

## Performance Benefits

UV provides significant performance improvements over pip:

- **Faster installations**: 10-100x faster package installation
- **Better dependency resolution**: More reliable resolution of package conflicts
- **Improved caching**: Intelligent caching reduces repeated downloads
- **Parallel downloads**: Multiple packages downloaded simultaneously
- **Better error messages**: More informative error reporting

## Migration from pip

UV is designed as a drop-in replacement for pip. Most pip commands work with `uv pip`:

```bash
# Old pip commands
pip install package
pip install -r requirements.txt
pip install -e .

# New UV commands
uv pip install package
uv pip install -r requirements.txt
uv pip install -e .
```

## CI/CD Integration

For GitHub Actions and CI systems, UV installation is straightforward:

```yaml
- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: |
    export PATH="$HOME/.local/bin:$PATH"
    uv sync --locked   # fails when uv.lock no longer matches pyproject.toml
```

## Troubleshooting

### Common Issues

1. **UV not found**: Ensure UV is in your PATH after installation
2. **Package conflicts**: UV's resolver is more strict than pip - this usually indicates real dependency issues
3. **Cache issues**: Clear UV cache with `uv cache clean` if needed

### Migrating from Conda/Mamba

If you're currently using conda/mamba (Option D from README):

```bash
# 1. Deactivate conda environment
conda deactivate

# 2. Create new UV-based environment and install ImSwitch at the locked versions
cd ~/Documents/ImSwitch
uv venv
uv sync

```

**Why migrate?**
- 10-100x faster package installation
- Better dependency resolution
- No need for conda/mamba overhead
- Native Python ecosystem tooling

### Getting Help

- UV Documentation: https://docs.astral.sh/uv/
- ImSwitch Issues: https://github.com/openUC2/ImSwitch/issues