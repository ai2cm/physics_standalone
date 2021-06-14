#!/bin/bash
set -e -x
scheduler_script=$1
echo "${JOB_NAME}-${BUILD_NUMBER}"

if [ $backend == 'numpy' ] && [ $physics == 'buildenv' ] && [ ! -d "venv/bin" ]
then
    python -m venv venv
    source venv/bin/activate
    git clone https://github.com/VulcanClimateModeling/gt4py.git
    pip install -e ./gt4py[cuda102]
    python -m gt4py.gt_src_manager install
    
else
    source ../../../../../numpy/physics/buildenv/slave/daint_submit/venv/bin/activate
    echo "venv already exists, continue without making one"
    echo $WORKSPACE
    echo `which python`
    echo `pip list`
    backend=${backend}
    phy=${physics}
    if [ $backend == 'gtcuda' ]
    then
        sed -i 's/<CPUSPERTASK>/1/g' ${scheduler_script}
    else
        sed -i 's/<CPUSPERTASK>/12/g' ${scheduler_script}
    fi
    sed -i -e "s/<which_backend>/${backend}/g" ${scheduler_script}
    sed -i -e "s/<which_physics>/${phy}/g" ${scheduler_script}
    export IS_DOCKER=False
    echo "Submitting slurm script:"
    cat ${scheduler_script}

    cp ${scheduler_script} runfile/.
    cd runfile
    OUT="${phy}_${backend}.out"
    set +e
    res=$(sbatch -W -C gpu ${scheduler_script} 2>&1)
    status1=$?
    grep -q SUCCESS ${OUT}
    status2=$?
    set -e
    wait
    echo "DONE WAITING ${status1} ${status2}"
    if [ $status1 -ne 0 -o $status2 -ne 0 ] ; then
    echo "ERROR: physics validation not successful"
    exit 1
    else
    echo "physics validation run successful"
    fi

    echo "Job completed!"
    # echo output of SLURM job
    cat ${OUT}
    rm ${OUT}
fi
