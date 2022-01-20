# ===================================
# Purpose: testing `physics_standalone`
# Stack:
#   - based on ubuntu20.04
#   - gcc-3.9.0
#   - python-3.8
#   - NCEP
#   - serialbox-2.6.1
#   - GT4Py (and GT v1)
#   - numpy, xarray
# ===================================
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt install -y --no-install-recommends \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa 

RUN apt-get update \
    && apt install -y --no-install-recommends \
    apt-utils \
    sudo \
    build-essential \
    gcc-9 \
    g++-9 \
    gfortran \
    gdb \
    wget \
    curl \
    tar \
    git \
    vim \
    nano \
    make \
    cmake \
    cmake-curses-gui \
    python3.8-dev \
    python3.8-distutils \
    libssl-dev \
    libboost-all-dev \
    libnetcdf-dev \
    libnetcdff-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10 && \
    update-alternatives  --set python /usr/bin/python3.8

# set TZ
ENV TZ=US/Pacific
RUN echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# install some python packages
RUN pip install numpy

# install serialbox from source
COPY serialbox /serialbox
RUN cd /serialbox && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/serialbox -DCMAKE_BUILD_TYPE=Debug \
    -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_ENABLE_FORTRAN=ON \
    -DSERIALBOX_TESTING=ON  ../ && \
    make -j4 && \
    make test && \
    make install && \
    /bin/rm -rf /serialbox
ENV PYTHONPATH=/usr/local/serialbox/python

# install gt4py
RUN cd /
RUN git clone -b v36 https://github.com/ai2cm/gt4py.git
RUN pip install ./gt4py && \
    python -m gt4py.gt_src_manager install -m 1

# install some python packages
RUN pip install numpy xarray[complete]

# add default user
ARG USER=user
ENV USER ${USER}
RUN useradd -ms /bin/bash ${USER} \
    && echo "${USER}   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
ENV USER_HOME /home/${USER}
RUN chown -R ${USER}:${USER} ${USER_HOME}

# create working directory
ARG WORKDIR=/work
ENV WORKDIR ${WORKDIR}
RUN mkdir ${WORKDIR}
RUN chown -R ${USER}:${USER} ${WORKDIR}

WORKDIR ${WORKDIR}
USER ${USER}

CMD ["/bin/bash"]

