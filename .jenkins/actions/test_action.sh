#!/bin/bash
set -e -x
scheduler_script=$1
echo "${JOB_NAME}-${BUILD_NUMBER}"
echo `pip list`
echo `which python`
echo `pwd`

sed -i 's|<NAME>|physics_standalone_validation|g'
sed -i 's|<NTASKS>|12\n#SBATCH \-\-hint=nomultithread|g' ${scheduler_script}
sed -i 's|00:45:00|00:30:00|g' ${scheduler_script}
sed -i 's|<OUTFILE>|out.log|g' ${scheduler_script}
sed -i 's|<NTASKSPERNODE>|1|g' ${scheduler_script}
sed -i 's/<CPUSPERTASK>/12/g' ${scheduler_script}

sed -i 's/<G2G>/export BACKEND=numpy/g' ${scheduler_script}
sed -i 's/<CMD>/python physics.py microph ../microph/data numpy None None/g' ${scheduler_script}
cat ${scheduler_script}