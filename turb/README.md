# GFS Turbulence Scheme -- GT4Py Porting

## Setup

### Docker

- `../build.sh` to build a Docker image
- `../enter.sh` to enter the container

### Piz Daint

The setup is identical to `stelliom_dev` microphysics.

```
source ./physics_standalone/turb/env_daint
python -m venv venv
source ./venv/bin/activate
git clone git@github.com:GridTools/gt4py.git
pip install -e ./gt4py[cuda102]
python -m gt4py.gt_src_manager install
pip install matplotlib
```

Confirm that the environments are sourced properly:

```
source ./physics_standalone/turb/env_daint
source ./venv/bin/activate
```

## Get input and reference data

If you have not retrieved the relevant data yet, execute `make get_data` 

## Run

Currently, only fortran version of the code is supported. Be sure to specify `VERSION=fortran`, GT4Py work in progress.

- Run for validation: `make validation VERSION=fortran`
- Run for normal benchmark: `make benchmark VERSION=fortran`