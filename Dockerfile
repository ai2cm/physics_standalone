FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y \
    apt-utils \
    sudo \
    build-essential \
    gcc \
    g++ \
    gfortran \
    gdb \
    wget \
    curl \
    tar \
    git \
    vim \
    make \
    cmake \
    cmake-curses-gui \
    python3-pip \
    python3-dev \
    libssl-dev \
    libboost-all-dev \
    libnetcdf-dev \
    libnetcdff-dev

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives  --set python /usr/bin/python3 && \
    update-alternatives  --set pip /usr/bin/pip3

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
    make -j8 && \
    make test && \
    make install && \
    /bin/rm -rf /serialbox

# Install Google Cloud
RUN apt-get update && apt-get install -y  apt-transport-https ca-certificates gnupg curl gettext

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list &&\
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -


RUN apt-get update && apt-get install -y google-cloud-sdk jq python3-dev python3-pip kubectl gfortran

# Zarr conversion
COPY serial_convert /serial_convert

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

# install some python packages
RUN pip install numpy zarr xarray ipython pyyaml dask netCDF4 rechunker && \
    pip install dask[array] --upgrade

WORKDIR ${WORKDIR}
USER ${USER}

CMD ["/bin/bash"]

