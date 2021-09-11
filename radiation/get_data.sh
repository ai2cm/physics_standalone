#!/bin/bash

if [ -z "$IS_DOCKER" ] ; then
    echo "Not in docker"
else
    gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output/LW /work/radiation/fortran/data/.
    gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz /work/radiation/python/lookupdata/.
    gsutil cp gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output/LW/* /work/radiation/fortran/radlw/dump/.

    cd /work/radiation/fortran/data/LW
    tar -xzvf data.tar.gz

    cd /work/radiation/python/lookupdata
    tar -xzvf lookup.tar.gz

    cd /work/radiation/fortran/radlw/dump
    tar -xzvf data.tar.gz
fi
