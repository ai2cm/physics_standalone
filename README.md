# FV3GFS Physical Parameterization Standalone Versions

This repo contains standalone versions of a selection of the FV3GFS physical parametrizations. They can be run using inputs that have been serialized from an actual c48 run of fv3gfs and are validated against serialized data from the same run right after the parameterization has been called.

## Requirements

You need the following things installed if you want to run directly on the host you are currently logged on to or if you are working directly on your laptop.
- serialbox installation (branch `savepoint_as_string` from https://github.com/VulcanClimateModeling/serialbox2/)
- `gfortran`
- `wget`

## Running

```bash
cd XXX
./get_data.sh
make
./main.x
```

Note: You may have to adapt the `Makefile` to point to your serialbox installation if it is not installed under `/usr/local/serialbox`.

## Docker environment

If you prefer to work in a Docker environment, simply type `./build.sh` to build a Docker image that contains everything you need and `./enter` to enter the image with the correct mounted directories to work on the standalone codes.

## Code coverage

In order to inspect the code coverage that FV3GFS actually has (and the data used to run the parameterizations as well) take a look at the code [coverage report](https://htmlpreview.github.io/?https://github.com/VulcanClimateModeling/physics_standalone/blob/master/coverage/index.html).

## Generating serialized data

```
git clone fv3gfs-fortran
cd fv3gfs-fortran
git pull
git checkout serialize_physics
./phys_build.sh
[take any rundir and copy it here]
./phys_run.sh
cd /rundir
export SER_ENV=TURB
./submit_job.sh
```
