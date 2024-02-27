FROM nvcr.io/nvidia/l4t-jetpack:r35.3.1
# nvcr.io/nvidia/l4t-jetpack:r36.2.0
# nvcr.io/nvidia/l4t-jetpack:r35.3.1

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all the dependencies needed to build and run
RUN apt update && apt install -y ca-certificates gpg wget software-properties-common
RUN add-apt-repository universe && add-apt-repository multiverse

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt update
RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || rm /usr/share/keyrings/kitware-archive-keyring.gpg
RUN apt install -y kitware-archive-keyring

RUN apt update && apt full-upgrade -y && apt install -y --no-install-recommends \
    tzdata \
    git \
    openssh-client \
    locales \
    sudo \
    nano \
    curl \
    build-essential \
    pkg-config \
    libgl1-mesa-dev \
    libsm6 \
    libice6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libdbus-1-3 \
    libusb-1.0-0-dev \
    zlib1g-dev \
    python3 \
    python3-dev \
    zip \
    unzip \
    python3-pip \
    cmake \
    cmake-curses-gui \
    vim \
    doxygen \
    cppcheck \
    gdb \
    clang-tidy \
    clang-format \
    flawfinder \
    valgrind \
    ffmpeg \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa \
    gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
    python3-gi python3-gi-cairo \
    libfontconfig1-dev libfreetype6-dev libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev \
    libxi-dev libxrender-dev libxcb1-dev libxcb-cursor-dev libxcb-glx0-dev libxcb-keysyms1-dev \
    libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev \
    libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev libxcb-util-dev \
    libxcb-xinerama0-dev libxcb-xkb-dev libxkbcommon-dev libxkbcommon-x11-dev \    
    #libvulkan\
    libkrb5-dev libuv1-dev libssl-dev \
    libgl-dev libglx-dev libopengl-dev \
    autoconf \
    libtool \
    software-properties-common \
    gnupg2 \
    dkms \
    libgtest-dev \
    libtbb-dev libtbb2 \
    && apt-get -qq clean

RUN apt install -y libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev libpython3-dev libavcodec-dev \
    libavformat-dev libswscale-dev libopenblas-dev protobuf-compiler libprotoc-dev libhdf5-103 libhdf5-dev \
    libspdlog-dev

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh \
     && /bin/bash ~/miniconda.sh -b -u -p /opt/conda \
     && rm ~/miniconda.sh \
     && /opt/conda/bin/conda init bash \
     && /opt/conda/bin/conda init zsh

ENV PATH=/opt/conda/bin/:$PATH

ENV PYTHON_VERSION=3.8

# mmpose env
RUN conda create -y --name mmpose python=${PYTHON_VERSION}

RUN conda init bash && echo "source activate mmpose" > ~/.bashrc

SHELL [ "/bin/bash", "--login", "-c" ]

# SHELL [ "conda", "run", "-n", "mmpose", "/bin/bash", "-c" ]

WORKDIR /mmlab

# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# RUN wget https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl -O torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# RUN pip install --no-cache torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

# https://forums.developer.nvidia.com/t/module-torch-distributed-has-no-attribute-reduceop/256581/7
RUN wget https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl -O torch-1.11.0-cp38-cp38-linux_aarch64.whl
RUN pip install --no-cache torch-1.11.0-cp38-cp38-linux_aarch64.whl

# PyTorch v2.0 - torchvision v0.15.1
# PyTorch v1.11 - torchvision v0.12.0
RUN pip install torchvision==0.12.0

# WORKDIR /torchV
# ENV BUILD_VERSION=0.12.0
# RUN git clone --branch v0.12.0 https://github.com/pytorch/vision torchvision
# RUN cd torchvision && pip install -e .
# WORKDIR /mmlab

# https://mmdeploy.readthedocs.io/en/latest/01-how-to-build/jetsons.html#prepare-tensorrt
RUN cp -r /usr/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/envs/mmpose/lib/python${PYTHON_VERSION}/site-packages/

ENV TENSORRT_DIR=/usr/include/aarch64-linux-gnu
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# TODO:? https://mmdeploy.readthedocs.io/en/latest/01-how-to-build/jetsons.html#install-mmcv

RUN pip install ninja numpy
RUN pip install -U openmim
RUN mim install mmengine

# mmyolo env
RUN conda create -y --name mmyolo --clone mmpose

RUN conda init bash && echo "source activate mmyolo" > ~/.bashrc

SHELL [ "/bin/bash", "--login", "-c" ]

# SHELL [ "conda", "run", "-n", "mmyolo", "/bin/bash", "-c" ]


# mmdeploy env
RUN conda create -y --name mmdeploy --clone mmpose

RUN conda init bash && echo "source activate mmdeploy" > ~/.bashrc

SHELL [ "/bin/bash", "--login", "-c" ]

RUN pip install onnx==1.10.0

# ONNX
RUN wget https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl -O onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
RUN pip install onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl

RUN pip install versioned-hdf5 pycuda

RUN git clone https://github.com/openppl-public/ppl.cv.git
WORKDIR /mmlab/ppl.cv
ENV PPLCV_DIR=/mmlab/ppl.cv
# RUN ./build.sh cuda

WORKDIR /mmlab
RUN git clone -b main --recursive https://github.com/open-mmlab/mmdeploy.git
ENV MMDEPLOY_DIR=/mmlab/mmdeploy

RUN git clone -b 3.x https://github.com/open-mmlab/mmdetection.git


# Needs GPU / nvidia runtime:
# RUN python3 -c "import torch;print(f'CUDA IS FOUND:{torch.cuda.is_available()}')"
# RUN mim install -v "mmcv>=2.0.1,<2.1.0"
# RUN mim install -v "mmdet>=3.1.0,<4.0.0"

# RUN git clone https://github.com/open-mmlab/mmpose.git
# RUN cd mmpose && pip install -r requirements.txt
# RUN cd mmpose && pip install -v -e .

# RUN git clone https://github.com/open-mmlab/mmyolo.git
# RUN cd mmyolo && pip install -r requirements.txt
# RUN cd mmyolo && pip install -v -e .
