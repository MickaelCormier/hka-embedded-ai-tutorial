FROM nvcr.io/nvidia/l4t-jetpack:r35.3.1
#nvcr.io/nvidia/l4t-base:r36.2.0

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
    # QT 6.5 dependencies from: https://doc.qt.io/qt-6/linux-requirements.html --->
    libfontconfig1-dev libfreetype6-dev libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev \
    libxi-dev libxrender-dev libxcb1-dev libxcb-cursor-dev libxcb-glx0-dev libxcb-keysyms1-dev \
    libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync-dev libxcb-xfixes0-dev \
    libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev libxcb-util-dev \
    libxcb-xinerama0-dev libxcb-xkb-dev libxkbcommon-dev libxkbcommon-x11-dev \    
    # qt6-base-dev libqt6concurrent6 libqt6core6 \
    # libqt6dbus6 libqt6gui6 libqt6network6 libqt6openglwidgets6 libqt6printsupport6 \
    # libqt6sql6 libqt6test6 libqt6widgets6 libqt6xml6 libvulkan-dev qmake6 qt6-base-dev-tools \
    # qt6-qpa-plugins \
    #libvulkan\
    libkrb5-dev libuv1-dev libssl-dev \
    libgl-dev libglx-dev libopengl-dev \
    # gRPC Deps --->
    autoconf \
    libtool \
    software-properties-common \
    gnupg2 \
    dkms \
    libgtest-dev \
    libtbb-dev libtbb12 libtbb2 libtbbmalloc2 \
    && apt-get -qq clean

ENV QT_QPA_PLATFORM="xcb"

RUN pip install requests ninja pytest pandas numpy
