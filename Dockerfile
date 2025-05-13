# Use the official PyTorch image as the base
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Set non-interactive mode for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone to Berlin
ENV TZ=Europe/Berlin

# Link timezone files to system
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    wget \
    unzip \
    libegl1 \
    libgl1 \
    libgomp1 \
    ninja-build \
    cmake \
    build-essential \
    libopenblas-dev \
    xterm \
    xauth \
    openssh-server \
    tmux \
    mate-desktop-environment-core \
    tzdata \
    iputils-ping \
    graphviz && \
    rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch3D from a prebuilt wheel
RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
    pytorch3d==0.7.8+pt2.4.1cu124

# Install additional Python dependencies
RUN pip install \
    numpy==1.26.2 \
    matplotlib==3.5.3 \
    pyyaml \
    wandb \
    pykitti \
    opencv-python \
    open3d \
    nuscenes-devkit \
    "truckscenes-devkit[all]" \
    torchinfo \
    torchview \
    graphviz \
    ipykernel \
    timm

# Install additional packages for PTv3 backbone
RUN pip install h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops plyfile termcolor \
    && pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.html \
    && pip install torch-geometric \
    && pip install spconv-cu124 \
    && pip install flash-attn


# Install chamfer_distance from GitHub
RUN pip install git+'https://github.com/otaheri/chamfer_distance'

# Reset DEBIAN_FRONTEND to prevent unexpected behavior
ENV DEBIAN_FRONTEND=dialog

# Set the default command to run the container
CMD ["/bin/bash"]
