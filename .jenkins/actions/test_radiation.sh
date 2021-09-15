#!/bin/bash

# This scripts provides a Jenkins action to run and validate the radiation scheme
# with all relevant GT4Py backends. Several environment variables (see below) are exptected
# to be set upon execution.

# 2021/09/24 Andrew Pauling, Allen AI, andrewp@allenai.org

# stop on all errors (also within a pipe-redirection)
set -e -x
set -o pipefail

# the following environment variables need to be set
#  scheme  -  the radiation scheme to test (SW or LW)
#                       (note: parameterization="buildenv" is used to setup environment)
#  backend           -  the GT4Py backend to run
if [ -z "${scheme}" ] ; then
    echo "ERROR: the variable 'scheme' needs to be set to run this Jenkins action"
    exit 1
fi
if [ -z "${backend}" ] ; then
    echo "ERROR: the variable 'backend' needs to be set to run this Jenkins action"
    exit 1
fi

echo "Information:"
echo "  JOB_NAME=${JOB_NAME}"
echo "  BUILD_NUMBER=${BUILD_NUMBER}"
echo "  WORKSPACE=${WORKSPACE}"
echo "  parameterization=${parameterization}"
echo "  backend=${backend}"
echo "  venv=${venv}"
echo ""

if [ "${backend}" == "numpy" ] && [ "${parameterization}" == "buildenv" ] ; then

    if [ -d "./venv" ] ; then
        echo "WARNING: local Python virtual environment already present, skipping!"
        exit 0
    fi

    # setup Python virtual environment
    python -m venv venv
    source venv/bin/activate
    git clone https://github.com/VulcanClimateModeling/gt4py.git
    pip install -e ./gt4py[cuda102]
    python -m gt4py.gt_src_manager install
    deactivate

else

    # copy the necessary serialized data and extract it
    mkdir data
    gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/fv3gfs-fortran-output data/.
    gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata data/.
    gsutil cp -r gs://vcm-fv3gfs-serialized-regression-data/physics/standalone-output data/.

    export HOME=`pwd`

    cd data/fv3gfs-fortran-output/${scheme}
    tar -xzvf data.tar.gz
    cd $HOME/data/lookupdata
    tar -xzvf lookup.tar.gz
    cd $HOME/data/standalone-output/${scheme}
    tar -xzvf data.tar.gz
    cd $HOME

    # build the Docker image
    source ./build.sh

    # submit SLURM job
    export IS_DOCKER=True
    export IS_TEST=True
    export BACKEND=${backend}

    docker run \
        --mount type=bind,source=`pwd`/data/fv3gfs-fortran-output/LW,target=/deployed/radiation/fortran/data/LW \
        --mount type=bind,source=`pwd`/data/lookupdata,target=/deployed/radiation/python/lookupdata \
        --mount type=bind,source=`pwd`/data/standalone-output/LW,target=/deployed/radiation/fortran/radlw/dump \
        --env IS_TEST=${IS_TEST} \
        --env IS_DOCKER=${IS_DOCKER} \
        --env BACKEND=${BACKEND} \
        physics_standalone /bin/bash -c 'cd /deployed/radiation/python/radlw && python test_${scheme}_gt4py.py'

fi
