# Use an appropriate base image for Jetson Nano
# sudo docker build -t imswitch_hik .
# sudo docker run -it --privileged  imswitch_hik
# sudo docker ps # => get id for stop
# docker stop imswitch_hik
# sudo docker inspect imswitch_hik
# docker run --privileged -it imswitch_hik
# For Raspberry Pi with Picamera2:
# docker run -it --privileged \
#   -v /run/dbus/system_bus_socket:/run/dbus/system_bus_socket \
#   -v /dev/dma_heap:/dev/dma_heap \
#   --device /dev/video0:/dev/video0 \
#   --device /dev/video10:/dev/video10 \
#   --device /dev/video11:/dev/video11 \
#   --device /dev/video12:/dev/video12 \
#   -e MODE=terminal \
#   ghcr.io/openuc2/imswitch-noqt:sha-5d54391
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e UPDATE_GIT=0 -e UPDATE_CONFIG=0 --privileged ghcr.io/openuc2/imswitch-noqt-x64:latest
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e UPDATE_GIT=1 -e UPDATE_CONFIG=0 --privileged imswitch_hik
# performs python3 /opt/MVS/Samples/aarch64/Python/MvImport/GrabImage.py
#  sudo docker run -it --entrypoint=/venv-shell.sh imswitch_hik
# docker build --build-arg ARCH=linux/arm64  -t imswitch_hik_arm64 .
# docker build --build-arg ARCH=linux/amd64  -t imswitch_hik_amd64 .
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e UPDATE_GIT=0 -e UPDATE_CONFIG=0 --privileged imswitch_hik
#
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e UPDATE_GIT=1 -e UPDATE_CONFIG=0 --privileged ghcr.io/openuc2/imswitch-noqt-x64:latest
# For loading external configs and store data externally
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -e HEADLESS=1  -e HTTP_PORT=8001 -e UPDATE_GIT=1 -e UPDATE_CONFIG=0 -e CONFIG_PATH=/config  --privileged  -v ~/Downloads:/config  imswitch_hik_arm64
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -e HEADLESS=1  -e HTTP_PORT=8001 -e UPDATE_GIT=1 -e UPDATE_CONFIG=0 --privileged -e DATA_PATH=/dataset  -v /media/uc2/SD2/:/dataset ghcr.io/openuc2/imswitch-noqt-x64:latest
# docker run -it --entrypoint=/venv-shell.sh ghcr.io/openuc2/imswitch-noqt-arm64:latest
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22  -e UPDATE_INSTALL_GIT=1  UC2-REST"  -e CONFIG_PATH=/Users/bene/Downloads  -e DATA_PATH=/Users/bene/Downloads  -v ~/Documents/imswitch_docker/imswitch_git:/tmp/ImSwitch-changes  -v ~/Documents/imswitch_docker/imswitch_pip:/persistent_pip_packages  -v /media/uc2/SD2/:/dataset  -v ~/Downloads:/config  --privileged imswitch_hik
# sudo docker pull docker pull ghcr.io/openuc2/imswitch-noqt-arm64:latest
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e UPDATE_GIT=0 -e UPDATE_CONFIG=0 --privileged imswitch_hik_arm64
# docker build -t ghcr.io/openuc2/imswitch-noqt-arm64:latest .
# docker build -t imswitch-holo .

# DOCKER_BUILDKIT=1 docker build -t imswitch-holo .
# sudo apt install docker-buildx-plugin


# Witht he following configuration we can do the following:
# 1. Update the ImSwitch repository and install the changes and make them persistent by mounting a volume to /tmp/ImSwitch-changes and /persistent_pip_packages respectively
# both of which are mounted to the host machine directories
# 2. Use a ImSwitchConfig folder that is mounted to the host machine directory /root/ImSwitchConfig
# 3. Use a dataset folder that is mounted to the host machine directory /media/uc2/SD2
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 \
# -e UPDATE_INSTALL_GIT=1 \
# -e DATA_PATH=/dataset \
# -e CONFIG_PATH=/config \
# -v ~/Documents/imswitch_docker/imswitch_git:/tmp/ImSwitch-changes \
# -v ~/Documents/imswitch_docker/imswitch_pip:/persistent_pip_packages \
# -v /media/uc2/SD2/:/dataset \
# -v ~/Downloads:/config


# Use an appropriate base image for multi-arch support
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Needed for platform detection in install-drivers.sh:
ARG TARGETPLATFORM

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install the project into `/opt/imswitch`
WORKDIR /opt/imswitch

# Keep Python from buffering stdout and stderr to avoid situations where the application crashes without emitting any logs due to buffering:
ENV PYTHONUNBUFFERED=1
# Enable bytecode compilation:
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since the source tree is a mounted volume:
ENV UV_LINK_MODE=copy
# Omit development dependencies
ENV UV_NO_DEV=1
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Note(ethanjli): we have RUN steps calling build scripts which each create and delete many files,
# in order to prevent junk from being baked into the final container image. This also gives us more
# helpful error messages when a particular command fails.
# We split up the work into different scripts run at different stages to facilitate correct
# container image caching.

# Install rarely-changing hardware drivers in a lower container image layer:
RUN \
  --mount=type=cache,sharing=locked,target=/var/cache/apt \
  --mount=type=cache,sharing=locked,target=/var/lib/apt \
  --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=docker/raspberrypi.gpg,target=/mnt/build/raspberrypi.gpg \
  --mount=type=bind,source=docker/install-drivers.sh,target=/mnt/build/install-drivers.sh \
  /mnt/build/install-drivers.sh
ENV MVCAM_COMMON_RUNENV=/opt/MVS/lib
# LD_LIBRARY_PATH is set separately; expanding $LD_LIBRARY_PATH before it is defined causes a
# Dockerfile linter warning and is a no-op at build time anyway.
ENV LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32
ENV GENICAM_GENTL64_PATH=/opt/VimbaX/cti

# Larger slowly-changing dependencies are installed in a separate container image layer before the
# rapidly-changing ImSwitch repository:
RUN \
  --mount=type=cache,sharing=locked,target=/var/cache/apt \
  --mount=type=cache,sharing=locked,target=/var/lib/apt \
  --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=docker/install-imswitch-deps.sh,target=/mnt/build/install-imswitch-deps.sh \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  /mnt/build/install-imswitch-deps.sh

# Install Imswitch itself:
COPY . /opt/imswitch
RUN \
  --mount=type=cache,sharing=locked,target=/var/cache/apt \
  --mount=type=cache,sharing=locked,target=/var/lib/apt \
  --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen

# Install UC2-REST itself (from latest uc2-rest from github master using: uv pip install https://github.com/openUC2/UC2-REST/archive/refs/heads/master.zip --force-reinstall)
RUN \
  --mount=type=cache,sharing=locked,target=/var/cache/apt \
  --mount=type=cache,sharing=locked,target=/var/lib/apt \
  --mount=type=cache,target=/root/.cache/uv \
  uv pip install https://github.com/openUC2/UC2-REST/archive/refs/heads/master.zip --force-reinstall

# Place executables in the environment at the front of the path
ENV PATH="/opt/imswitch/.venv/bin:$PATH"
# Expose HTTP port and Jupyter server port
EXPOSE 8001 8888 8889

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Run as unprivileged user
RUN \
  groupadd --system --gid 1000 pi && \
  groupadd --system --gid 989 spi && \
  groupadd --system --gid 988 i2c && \
  groupadd --system --gid 986 gpio && \
  useradd --system --gid 1000 --uid 1000 --create-home pi
USER pi
