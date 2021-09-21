#!/bin/bash

if [ -z "$IS_DOCKER" ] ; then
    echo "Not in docker"
else
    if [ -z "$(ls -A /work/radiation/fortran/data/LW)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/LW /work/radiation/fortran/data/.
        cd /work/radiation/fortran/data/LW
        tar -xzvf data.tar.gz
    else
        echo "Data already present"
    fi

    if [ -z "$(ls -A /work/radiation/python/lookupdata)" ]; then
        gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz /work/radiation/python/lookupdata/.
        cd /work/radiation/python/lookupdata
        tar -xzvf lookup.tar.gz
    else
        echo "Data already present"
    fi

    if [ -z "$(ls -A /work/radiation/fortran/radlw/dump)" ]; then
        gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/LW/* /work/radiation/fortran/radlw/dump/.
        cd /work/radiation/fortran/radlw/dump
        tar -xzvf data.tar.gz
    else
        echo "Data already present"
    fi
        
fi
