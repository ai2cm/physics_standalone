#!/bin/bash
set -e -x
scheduler_script=$1
echo "${JOB_NAME}-${BUILD_NUMBER}"

python -m venv venv
source ./venv/bin/activate
git clone https://github.com/VulcanClimateModeling/gt4py.git
pip install -e ./gt4py[cuda102]
python -m gt4py.gt_src_manager install

echo `which python`
echo `pip list`
backend=numpy
phy=seaice
sed -i 's/<CPUSPERTASK>/12/g' ${scheduler_script}
sed -i -e "s/<which_backend>/${backend}/g" ${scheduler_script}
sed -i -e "s/<which_physics>/${phy}/g" ${scheduler_script}
echo "Submitting slurm script:"
cat ${scheduler_script}

# submit SLURM job
launch_job ${scheduler_script} 9000
if [ $? -ne 0 ] ; then
    exitError 1251 ${LINENO} "problem launching SLURM job ${scheduler_script}"
fi

# echo output of SLURM job
OUT="${phy}_${backend}.out"
cat ${OUT}
rm ${OUT}
