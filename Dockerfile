FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt install -y --no-install-recommends \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa 

RUN apt-get update \
  && apt-get install -y --no-install-recommends\
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
    python3.9-dev \
    python3.9-distutils \
    libssl-dev \
    libboost-all-dev \
    libnetcdf-dev \
    libnetcdff-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10 && \
    update-alternatives  --set python /usr/bin/python3.9

# download and install NCEP libraries
RUN git config --global http.sslverify false && \
    git clone https://github.com/NCAR/NCEPlibs.git && \
    mkdir /opt/NCEPlibs && \
    cd NCEPlibs && \
    git checkout 3da51e139d5cd731c9fc27f39d88cb4e1328212b && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1

# set TZ
ENV TZ=US/Pacific
RUN echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

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

# install gt4py
RUN cd /
RUN git clone -b v35 https://github.com/VulcanClimateModeling/gt4py.git
RUN pip install -e ./gt4py && \
    python -m gt4py.gt_src_manager install -m 2

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

COPY --chown ${USER}:${USER} radiation /work/

WORKDIR ${WORKDIR}
USER ${USER}

CMD ["/bin/bash"]

