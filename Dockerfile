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
# Note: Debian Bookworm has Python 3.11 which is required for picamera2 compatibility
FROM debian:bookworm

ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Note(ethanjli): we have RUN steps calling build scripts which each create and delete many files,
# in order to prevent junk from being baked into the final container image. This also gives us more
# helpful error messages when a particular command fails.
# We split up the work into different scripts run at different stages to facilitate correct
# container image caching.

RUN --mount=type=bind,source=docker/build-conda.sh,target=/mnt/build/build-conda.sh /mnt/build/build-conda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN --mount=type=bind,source=docker/build-drivers.sh,target=/mnt/build/build-drivers.sh /mnt/build/build-drivers.sh
ENV MVCAM_COMMON_RUNENV=/opt/MVS/lib LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:"$LD_LIBRARY_PATH"
ENV GENICAM_GENTL64_PATH="/opt/VimbaX/cti"

# Larger slowly-changing dependencies are installed in a separate container image layer before the
# rapidly-changing ImSwitch repository:
RUN --mount=type=bind,source=docker/build-imswitch-deps.sh,target=/mnt/build/build-imswitch-deps.sh --mount=type=bind,source=./pyproject.toml,target=/mnt/ImSwitch/pyproject.toml /mnt/build/build-imswitch-deps.sh

# Always pull the latest version of ImSwitch and UC2-REST repositories
# Question(ethanjli): if we're copying the ImSwitch & UC2-REST repositories from local files using
# the COPY directive, shouldn't that ignore the cache anyways? Is there any way we can get rid of
# this BUILD_DATE hack?
# Adding a dynamic build argument to prevent caching
ARG BUILD_DATE
RUN --mount=type=bind,source=docker/build-imswitch.sh,target=/mnt/build/build-imswitch.sh --mount=type=bind,source=.,target=/mnt/ImSwitch /mnt/build/build-imswitch.sh
# Expose HTTP port and Jupyter server port
EXPOSE 8001 8888 8889

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/venv-shell.sh /venv-shell.sh
RUN chmod +x /entrypoint.sh /venv-shell.sh

ENTRYPOINT ["/entrypoint.sh"]
