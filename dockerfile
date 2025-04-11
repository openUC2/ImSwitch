# Use an appropriate base image for Jetson Nano
# sudo docker build -t imswitch_hik .
# sudo docker run -it --privileged  imswitch_hik
# sudo docker ps # => get id for stop
# docker stop imswitch_hik
# sudo docker inspect imswitch_hik
# docker run --privileged -it imswitch_hik
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e CONFIG_FILE=example_virtual_microscope.json -e UPDATE_GIT=0 -e UPDATE_CONFIG=0 --privileged ghcr.io/openuc2/imswitch-noqt-x64:latest
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e CONFIG_FILE=example_uc2_hik_flowstop.json -e UPDATE_GIT=1 -e UPDATE_CONFIG=0 --privileged imswitch_hik
# performs python3 /opt/MVS/Samples/aarch64/Python/MvImport/GrabImage.py
#  sudo docker run -it -e MODE=terminal imswitch_hik
# docker build --build-arg ARCH=linux/arm64  -t imswitch_hik_arm64 .
# docker build --build-arg ARCH=linux/amd64  -t imswitch_hik_amd64 .
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e CONFIG_FILE=example_virtual_microscope.json -e UPDATE_GIT=0 -e UPDATE_CONFIG=0 --privileged imswitch_hik
#
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -e CONFIG_FILE=example_uc2_hik_flowstop.json -e UPDATE_GIT=1 -e UPDATE_CONFIG=0 --privileged ghcr.io/openuc2/imswitch-noqt-x64:latest
# For loading external configs and store data externally
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -e HEADLESS=1  -e HTTP_PORT=8001    -e UPDATE_GIT=1  -e UPDATE_CONFIG=0  -e CONFIG_PATH=/config  --privileged  -v ~/Downloads:/config  imswitch_hik_arm64
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -e HEADLESS=1  -e HTTP_PORT=8001  -e UPDATE_GIT=1  -e UPDATE_CONFIG=0  --privileged -e DATA_PATH=/dataset  -v /media/uc2/SD2/:/dataset -e CONFIG_FILE=example_uc2_hik_flowstop.json ghcr.io/openuc2/imswitch-noqt-x64:latest
# docker run -it -e MODE=terminal ghcr.io/openuc2/imswitch-noqt-arm64:latest
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22  -e UPDATE_INSTALL_GIT=1  -e PIP_PACKAGES="arkitekt UC2-REST"  -e CONFIG_PATH=/Users/bene/Downloads  -e DATA_PATH=/Users/bene/Downloads  -v ~/Documents/imswitch_docker/imswitch_git:/tmp/ImSwitch-changes  -v ~/Documents/imswitch_docker/imswitch_pip:/persistent_pip_packages  -v /media/uc2/SD2/:/dataset  -v ~/Downloads:/config  --privileged imswitch_hik
# sudo docker pull docker pull ghcr.io/openuc2/imswitch-noqt-x64:latest

# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 -p 2222:22 -e HEADLESS=1 -e HTTP_PORT=8001 -v ~:/config  -e CONFIG_PATH=/config -e CONFIG_FILE=example_uc2_vimba.json -e UPDATE_GIT=0 -e UPDATE_CONFIG=0 --privileged ghcr.io/openuc2/imswitch-noqt-x64:latest



# Witht he following configuration we can do the following:
# 1. Update the ImSwitch repository and install the changes and make them persistent by mounting a volume to /tmp/ImSwitch-changes and /persistent_pip_packages respectively
# both of which are mounted to the host machine directories
# 2. Use a ImSwitchConfig folder that is mounted to the host machine directory /root/ImSwitchConfig
# 3. Use a dataset folder that is mounted to the host machine directory /media/uc2/SD2
# 4. Install additional pip packages by setting the PIP_PACKAGES environment variable to a space separated list of packages and make them persistent by mounting a volume to /persistent_pip_packages
# sudo docker run -it --rm -p 8001:8001 -p 8002:8002 \
# -e UPDATE_INSTALL_GIT=1 \
# -e PIP_PACKAGES="arkitekt UC2-REST" imswitch_hik \
# -e DATA_PATH=/dataset \
# -e CONFIG_PATH=/config \
# -v ~/Documents/imswitch_docker/imswitch_git:/tmp/ImSwitch-changes \
# -v ~/Documents/imswitch_docker/imswitch_pip:/persistent_pip_packages \
# -v /media/uc2/SD2/:/dataset \
# -v ~/Downloads:/config 


# Use an appropriate base image for multi-arch support
FROM ubuntu:22.04

ARG TARGETPLATFORM
ENV TZ=America/Los_Angeles

# Install necessary dependencies and prepare the environment as usual
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    python3 \
    python3-pip \
    build-essential \
    git \
    mesa-utils \
    libhdf5-dev \
    nano \
    usbutils \
    sudo \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install Miniforge based on architecture
RUN if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
        wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh; \
    fi && \
    /bin/bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Update PATH environment variable
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment and install packages
RUN /opt/conda/bin/conda create -y --name imswitch python=3.11 && \
    /opt/conda/bin/conda install -n imswitch -y -c conda-forge h5py numcodecs && \
    /opt/conda/bin/conda clean --all -f -y && \
    rm -rf /opt/conda/pkgs/*

# Download and install the appropriate Hik driver based on architecture
RUN cd /tmp && \
    wget https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_GML_V2.1.2_231116.zip && \
    unzip MVS_STD_GML_V2.1.2_231116.zip && \
    if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
        dpkg -i MVS-2.1.2_aarch64_20231116.deb && \
        rm -f MVS-2.1.2_aarch64_20231116.deb; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        dpkg -i MVS-2.1.2_x86_64_20231116.deb && \
        rm -f MVS-2.1.2_x86_64_20231116.deb; \
    fi && \
    rm -rf /tmp/MVS_STD_GML_V2.1.2_231116.zip /tmp/MVS_STD_GML_V2.1.2_231116/

## Install Daheng Camera 
# Create the udev rules directory
RUN mkdir -p /etc/udev/rules.d

# Download and install the appropriate Daheng driver based on architecture
RUN cd /tmp && \ 
wget https://dahengimaging.com/downloads/Galaxy_Linux_Python_2.0.2106.9041.tar_1.gz && \
tar -zxvf Galaxy_Linux_Python_2.0.2106.9041.tar_1.gz && \
if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
    wget https://dahengimaging.com/downloads/Galaxy_Linux-armhf_Gige-U3_32bits-64bits_1.5.2303.9202.zip && \
    unzip Galaxy_Linux-armhf_Gige-U3_32bits-64bits_1.5.2303.9202.zip && \
    cd /tmp/Galaxy_Linux-armhf_Gige-U3_32bits-64bits_1.5.2303.9202; \
elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
    wget https://dahengimaging.com/downloads/Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221.zip && \
    unzip Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221.zip && \
    cd /tmp/Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221; \
fi && \
chmod +x Galaxy_camera.run && \
cd /tmp/Galaxy_Linux_Python_2.0.2106.9041/api && \
/bin/bash -c "source /opt/conda/bin/activate imswitch && python3 setup.py build" && \
python3 setup.py install

# Run the installer script using expect to automate Enter key presses
RUN if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
    echo "Y En Y" | /tmp/Galaxy_Linux-armhf_Gige-U3_32bits-64bits_1.5.2303.9202/Galaxy_camera.run; \
elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
    echo "Y En Y" | /tmp/Galaxy_Linux-x86_Gige-U3_32bits-64bits_1.5.2303.9221/Galaxy_camera.run; \
fi

# Ensure the library path is set
ENV LD_LIBRARY_PATH="/usr/lib:/tmp/Galaxy_Linux-armhf_Gige-U3_32bits-64bits_1.5.2303.9202:$LD_LIBRARY_PATH"

# Source the bashrc file
ENV PATH=/opt/conda/bin:$PATH
# RUN echo "source ~/.bashrc" >> ~/.bashrc
# RUN /bin/bash -c "source ~/.bashrc"
RUN mkdir -p /opt/MVS/bin/fonts

# Set environment variable for MVCAM_COMMON_RUNENV
ENV MVCAM_COMMON_RUNENV=/opt/MVS/lib LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:$LD_LIBRARY_PATH


# install numcodecs via conda
RUN /opt/conda/bin/conda install numcodecs=0.15.0
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && \
    conda install scikit-image=0.19.3 -c conda-forge"
    

# fix the version of OME-ZARR 
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install ome-zarr==0.9.0"
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install numpy==1.26.4"


# Install UC2-REST first - as it will be installed via ImSwitch again
RUN git clone https://github.com/openUC2/UC2-REST /tmp/UC2-REST && \
    cd /tmp/UC2-REST && \
    /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"


# first install all the dependencies not not to install them again in a potential "breaking update"
# Clone the repository and install dependencies
RUN git clone https://github.com/openUC2/imSwitch /tmp/ImSwitch && \
    cd /tmp/ImSwitch && \
    /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Clone the config folder
RUN git clone https://github.com/openUC2/ImSwitchConfig /root/ImSwitchConfig

# we want psygnal to be installed without binaries - so first remove it 
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip uninstall psygnal -y"
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install psygnal --no-binary :all:"

# fix the version of OME-ZARR 
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install zarr==2.11.3"
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install ome-zarr==0.9.0"
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install numpy==1.26.4"


# TMP: Install Vimba 
# Download, extract, and install Vimba
RUN wget https://downloads.alliedvision.com/Vimba_v6.0_ARM64.tgz -O /tmp/Vimba_arm64.tgz && \
tar -xzf /tmp/Vimba_arm64.tgz -C /opt && \
rm /tmp/Vimba_arm64.tgz && \
cd /opt/Vimba_6_0/VimbaUSBTL && \
./Install.sh 

# Install Python bindings and VimbaPython   
RUN cd /opt/Vimba_6_0/VimbaPython/Source && \
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install ." && \
/bin/bash -c "source /opt/conda/bin/activate imswitch && pip install https://github.com/alliedvision/VimbaPython/archive/refs/heads/master.zip"

# Set environment variable for GenTL detection
ENV GENICAM_GENTL64_PATH="${GENICAM_GENTL64_PATH}:/opt/Vimba_6_0/VimbaUSBTL/CTI/arm_64bit"

# install IOHub - as it will be installed via ImSwitch again
#         "iohub @ https://github.com/czbiohub-sf/iohub/archive/refs/heads/main.zip"
RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install --no-cache-dir Cython"
RUN git clone https://github.com/czbiohub-sf/iohub /root/iohub && \
    cd /root/iohub && \
    /bin/bash -c "source /opt/conda/bin/activate imswitch && \
                  pip install --no-build-isolation /root/iohub"

# Install and configure OpenSSH
RUN apt-get update && apt-get install -y openssh-server && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create run directory for SSH
RUN mkdir /var/run/sshd

# Permit root login via SSH (not recommended for production by default)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config


RUN /bin/bash -c "source /opt/conda/bin/activate imswitch && \
    pip install dataclasses_json==0.5.7 typing_extensions==4.5.0"
    
# Always pull the latest version of ImSwitch and UC2-REST repositories
# Adding a dynamic build argument to prevent caching
ARG BUILD_DATE
RUN echo "Building on ${BUILD_DATE}"

# Clone the config folder
RUN cd /root/ImSwitchConfig && \
    git pull

# now update potential breaking changes
RUN cd /tmp/ImSwitch && \
    git pull && \
    /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/ImSwitch"

# Install UC2-REST
RUN cd /tmp/UC2-REST && \
    git pull && \
    /bin/bash -c "source /opt/conda/bin/activate imswitch && pip install /tmp/UC2-REST"

    
# Expose FTP port and HTTP port
EXPOSE  8001 8002 8003 8888 8889 22

ADD docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"] 