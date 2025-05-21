# Base image with CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TMPDIR=/var/tmp/
ENV COPPELIASIM_ROOT=/opt/coppeliaSim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PYTHONPATH=/root/install/RVT:\
/root/install/RVT/rvt/libs/PyRep:\
/root/install/RVT/rvt/libs/RLBench:\
/root/install/RVT/rvt/libs/YARR:\
/root/install/RVT/rvt/libs/peract_colab:\
/root/install/RVT/rvt/libs/point-renderer:\
$PYTHONPATH

# Set up temporary directories
RUN chmod 1777 /tmp && mkdir -p /var/tmp && chmod 1777 /var/tmp

# Install essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common curl wget unzip ca-certificates gnupg && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/*

# NEEDED FOR HEADLESS
RUN apt update && apt install xvfb xserver-xorg xinit nvidia-settings kmod -y
RUN apt update && apt install -y libglu1-mesa-dev mesa-utils xterm xauth x11-xkb-utils xfonts-base xkb-data libxtst6 libxv1
RUN apt install libxkbcommon-x11-0 -y

# Install development tools and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common build-essential cmake git curl wget unzip \
    ca-certificates gnupg python3.8 python3.8-dev python3.8-venv python3-pip

# Set Python 3.8 as default
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set up Python environment
RUN python -m pip install --upgrade "pip<24.1" setuptools

# Download and install CoppeliaSim
WORKDIR /tmp
RUN wget -O coppelia_player.tar.xz https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu20_04.tar.xz && \
    mkdir -p /opt/coppeliaSim && \
    tar -xJf coppelia_player.tar.xz -C /opt/coppeliaSim --strip-components=1 && \
    chmod +x /opt/coppeliaSim/coppeliaSim.sh && \
    rm coppelia_player.tar.xz

# Set CoppeliaSim environment variables
ENV COPPELIASIM_ROOT=/opt/coppeliaSim
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# Clone RVT repo with submodules
RUN git clone --recurse-submodules https://github.com/NVlabs/RVT.git /root/install/RVT && \
    cd /root/install/RVT && git submodule update --init

# Install PyTorch with CUDA support
RUN python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install additional Python packages
RUN python -m pip install yacs omegaconf hydra-core pandas PyYAML

# Set up environment for PyTorch3D
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"

# Install PyTorch3D dependencies and package
RUN python -m pip install iopath fvcore && \
    pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# Install RVT and its components
WORKDIR /root/install/RVT
RUN python -m pip install -e '.[xformers]' || pip install -e .

# Install required libraries
RUN python -m pip install -e rvt/libs/PyRep && \
    python -m pip install -e rvt/libs/RLBench && \
    python -m pip install -e rvt/libs/YARR && \
    python -m pip install -e rvt/libs/peract_colab && \
    python -m pip install -e rvt/libs/point-renderer

# Replace OpenCV with headless version
RUN pip uninstall -y opencv-python && \
    pip install opencv-python-headless

# Run sanity check
RUN python -c "print('✅ RVT2 environment successfully built.')"

# Set working directory
WORKDIR /root/install/RVT

# Default command
CMD ["/bin/bash"]