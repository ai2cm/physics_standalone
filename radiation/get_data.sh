#!/bin/bash

if [ ! -z "$IS_DOCKER" ] ; then
    echo "This script cannot be run in the Docker image"
else

    export HOME=`pwd`

    if [ ! -d "./fortran/data" ]; then
        cd ./fortran
        mkdir data
        cd data
        mkdir LW
        mkdir SW
        cd $HOME
    else
        echo "Fortran output directory already exists"
    fi

    if [ -z "$(ls -A ./fortran/data/LW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/radlw/* ./fortran/data/LW/.
        cd ./fortran/data/LW
        tar -xzvf data.tar.gz
        cd $HOME
    else
        echo "LW Fortran data already present"
    fi

    if [ -z "$(ls -A ./fortran/data/SW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/radsw/* ./fortran/data/SW/.
        cd ./fortran/data/SW
        tar -xzvf data.tar.gz
        cd $HOME
    else
        echo "SW Fortran data already present"
    fi

    if [ -z "$(ls -A ./python/lookupdata)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz ./python/lookupdata/.
        cd ./python/lookupdata
        tar -xzvf lookup.tar.gz
        cd $HOME
    else
        echo "Data already present"
    fi

    if [ ! -d "./fortran/radlw/dump" ]; then
        cd ./fortran/radlw
        mkdir dump
        cd $HOME
    else
        echo "LW standalone output directory already exists"
    fi

    if [ ! -d "./fortran/radsw/dump" ]; then
        cd ./fortran/radsw
        mkdir dump
        cd $HOME
    else
        echo "SW standalone output directory already exists"
    fi

    if [ -z "$(ls -A ./fortran/radlw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/radlw/* ./fortran/radlw/dump/.
        cd ./fortran/radlw/dump
        tar -xzvf data.tar.gz
        cd $HOME
    else
        echo "LW standalone data already present"
    fi

    if [ -z "$(ls -A ./fortran/radsw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/radsw/* ./fortran/radsw/dump/.
        cd ./fortran/radsw/dump
        tar -xzvf data.tar.gz
        cd $HOME
    else
        echo "SW standalone data already present"
    fi

    if [ ! -d "./python/forcing" ]; then
        cd ./python
        mkdir forcing
        cd $HOME
    else
        echo "Forcing directory already exists"
    fi

    if [ -z "$(ls -A ./python/forcing)" ]; then
	    gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/* ./python/forcing/.
	    cd ./python/forcing
	    tar -xzvf data.tar.gz
        cd $HOME
    else
	    echo "Forcing data already present"
    fi  
fi
