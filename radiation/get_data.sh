#!/bin/bash

if [ -z "$IS_DOCKER" ] ; then
    echo "Not in docker, please build and enter the Docker image before running this script"
else

    if [ ! -d "/work/radiation/fortran/data"]; then
        cd /work/radiation/fortran
        mkdir data
        cd /work/radiation
    else
        echo "Fortran output directory already exists"
    fi

    if [ -z "$(ls -A /work/radiation/fortran/data/LW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/radlw /work/radiation/fortran/data/.
        cd /work/radiation/fortran/data/LW
        tar -xzvf data.tar.gz
    else
        echo "LW Fortran data already present"
    fi

    if [ -z "$(ls -A /work/radiation/fortran/data/SW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/radsw /work/radiation/fortran/data/.
        cd /work/radiation/fortran/data/SW
        tar -xzvf data.tar.gz
    else
        echo "SW Fortran data already present"
    fi

    if [ -z "$(ls -A /work/radiation/python/lookupdata)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz /work/radiation/python/lookupdata/.
        cd /work/radiation/python/lookupdata
        tar -xzvf lookup.tar.gz
    else
        echo "Data already present"
    fi

    if [ ! -d "/work/radiation/fortran/radlw/dump"]; then
        cd /work/radiation/fortran/radlw
        mkdir dump
        cd /work/radiation
    else
        echo "LW standalone output directory already exists"
    fi

    if [ ! -d "/work/radiation/fortran/radsw/dump"]; then
        cd /work/radiation/fortran/radsw
        mkdir dump
        cd /work/radiation
    else
        echo "SW standalone output directory already exists"
    fi

    if [ -z "$(ls -A /work/radiation/fortran/radlw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/radlw/* /work/radiation/fortran/radlw/dump/.
        cd /work/radiation/fortran/radlw/dump
        tar -xzvf data.tar.gz
    else
        echo "LW standalone data already present"
    fi

    if [ -z "$(ls -A /work/radiation/fortran/radsw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/radsw/* /work/radiation/fortran/radsw/dump/.
        cd /work/radiation/fortran/radsw/dump
        tar -xzvf data.tar.gz
    else
        echo "SW standalone data already present"
    fi

    if [ ! -d "/work/radiation/python/forcing"]; then
        cd /work/radiation/python
        mkdir forcing
        cd ../
    else
        echo "Forcing directory already exists"
    fi

    if [ -z "$(ls -A /work/radiation/fortran/radlw/dump)" ]; then
	    gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/* /work/radiation/python/forcing/.
	    cd /work/radiation/python/forcing
	    tar -xzvf data.tar.gz
    else
	    echo "Forcing data already present"
    fi  
fi
