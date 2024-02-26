FROM dustynv/gstreamer:r35.3.1
#nvcr.io/nvidia/l4t-jetpack:r35.3.1
#nvcr.io/nvidia/l4t-base:r36.2.0

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all the dependencies needed to build and run
RUN apt update && apt install -y ca-certificates gpg wget software-properties-common
RUN add-apt-repository universe && add-apt-repository multiverse

RUN apt update

RUN apt install -y --no-install-recommends \
    tzdata \
    git \
    openssh-client \
    locales \
    sudo \
    nano \
    curl \
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
    zip \
    unzip \
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
    gstreamer1.0-tools gstreamer1.0-x \
    gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
    python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
    libfontconfig1-dev libfreetype6-dev libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev \
    libxi-dev libxrender-dev libxcb1-dev libxcb-cursor-dev libxcb-glx0-dev libxcb-keysyms1-dev \
    libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev \
    libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev libxcb-util-dev \
    libxcb-xinerama0-dev libxcb-xkb-dev libxkbcommon-dev libxkbcommon-x11-dev \
    libkrb5-dev libuv1-dev libssl-dev \
    libgl-dev \
    libtbb-dev libtbb2 \
    && apt-get -qq clean

ENV QT_QPA_PLATFORM="xcb"

RUN pip install requests ninja pytest pandas numpy
