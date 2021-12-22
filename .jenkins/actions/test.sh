#!/bin/bash

# This scripts provides a Jenkins action to run and validate all physical parameterizations
# with all relevant GT4Py backends. Several environment variables (see below) are exptected
# to be set upon execution.

# 2021/06/21 Elynn Wu, Vulcan Inc, elynnw@vulcan.com

# stop on all errors (also within a pipe-redirection)
set -e -x
set -o pipefail

# the following environment variables need to be set
#  parameterization  -  the physical parameterization to test
#                       (note: parameterization="buildenv" is used to setup environment)
#  backend           -  the GT4Py backend to run
if [ -z "${parameterization}" ] ; then
    echo "ERROR: the variable 'parameterization' needs to be set to run this Jenkins action"
    exit 1
fi
if [ -z "${backend}" ] ; then
    echo "ERROR: the variable 'backend' needs to be set to run this Jenkins action"
    exit 1
fi

# GTC backend name fix: passed as gtc_gt_* but their real name are gtc:gt:*
#                       OR gtc_* but their real name is gtc:*
if [[ $backend = gtc_gt_* ]] ; then
    # sed explained: replace _ with :, two times
    backend=`echo $backend | sed 's/_/:/;s/_/:/'`
fi
if [[ $backend = gtc_* ]] ; then
    # sed explained: replace _ with :
    backend=`echo $backend | sed 's/_/:/'`
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

    # source the pre-built Python virtual environment
    venv="../../../../../numpy/parameterization/buildenv/slave/daint_submit/venv"
    if [ ! -f "${venv}/bin/activate" ] ; then
        if [ ! -f "./venv/bin/activate" ] ; then
            echo "ERROR: Virtual environment not accessible or corrupt (${venv})"
            exit 1
        else
            echo "WARNING: Using locally built virtual environemnt (./venv)"
            venv="./venv"
        fi
    fi
    source ${venv}/bin/activate

    cd ./runfile

    # make a modifiable copy of SLURM job
    scheduler_script="run_${parameterization}_${backend}.job"
    job_output=run_${parameterization}_${backend}.out
    cp ./run.job ./${scheduler_script}

    # set configurable parameters in SLURM job
    if [ $backend == 'gtcuda' ] ; then
        sed -i 's/<CPUSPERTASK>/1/g' ${scheduler_script}
    else
        sed -i 's/<CPUSPERTASK>/12/g' ${scheduler_script}
    fi
    sed -i -e "s|<OUTPUT>|${job_output}|g" ${scheduler_script}
    sed -i -e "s|<BACKEND>|${backend}|g" ${scheduler_script}
    sed -i -e "s|<PARAMETERIZATION>|${parameterization}|g" ${scheduler_script}
    sed -i -e "s|<OPTIONS>|--data_dir=/project/s1053/physics_standalone_serialized_test_data/c48/${parameterization}|g" ${scheduler_script}
    if [ "${parameterization}" == 'turb' ] ; then
        sed -i -e "s|00:30:00|00:45:00|g" ${scheduler_script}
    fi
    if [ "${parameterization}" == 'turb' ] && [ "${backend}" == 'gtcuda' ] ; then
        sed -i -e "s|00:30:00|02:15:00|g" ${scheduler_script}
    fi
    echo "====== SLURM job ======="
    cat ${scheduler_script}
    echo "========================"

    # submit SLURM job
    export IS_DOCKER=False
    set +e
    res=$(sbatch -W -C gpu ${scheduler_script} 2>&1)
    status1=$?
    grep -q SUCCESS ${job_output}
    status2=$?
    set -e
    wait
    echo "DONE WAITING ${status1} ${status2}"
    echo "=== SLURM job output ==="
    cat ${job_output}
    echo "========================"
    if [ "${status1}" -ne 0 -o "${status2}" -ne 0 ] ; then
        echo "ERROR: Parameterization validation not successful"
        exit 1
    fi

    deactivate
    echo "Parameterization validation successful!"

fi
