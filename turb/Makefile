STENCILS = normal

ifeq (${HYPERTH}, True)
	NODES = 1
	T_PER_CORE = 2
else
	NODES = 1
	T_PER_CORE = 1
endif

MAX_REPS = 10

VFLAGS = RUN_MODE="validation" OMP_NUM_THREADS=12 REPS=0
BFLAGS = RUN_MODE="normal" OMP_NUM_THREADS=12 REPS=${MAX_REPS}
WFLAGS = RUN_MODE="weak" REPS=${MAX_REPS}
SFLAGS = RUN_MODE="strong" REPS=${MAX_REPS}

ifeq (${IS_DOCKER}, False)
	FT_DFLAGS = OPT="-DREPS=0 -DWRITE_DATA -DPROGRESS_MODE -DEXACT_SVP -DUSE_MIN_EVAP -DPRINT_NML"
	FT_VFLAGS = OPT="-DRUN_MODE=0 -DREPS=0 -DUSE_MIN_EVAP -DPRINT_NML"
	FT_BFLAGS = OPT="-DRUN_MODE=1 -DREPS=${MAX_REPS} -DEXACT_SVP -DUSE_MIN_EVAP"
	FT_WFLAGS = OPT="-DRUN_MODE=2 -DREPS=${MAX_REPS} -DEXACT_SVP -DUSE_MIN_EVAP"
	FT_SFLAGS = OPT="-DRUN_MODE=3 -DREPS=${MAX_REPS} -DEXACT_SVP -DUSE_MIN_EVAP"
else
	FT_DFLAGS = OPT="-DREPS=0 -DWRITE_DATA -DPROGRESS_MODE -DEXACT_SVP -DUSE_MIN_EVAP -DPRINT_NML -DUSE_MPIF90"
	FT_VFLAGS = OPT="-DRUN_MODE=0 -DREPS=0 -DUSE_MIN_EVAP -DPRINT_NML -DUSE_MPIF90"
	FT_BFLAGS = OPT="-DRUN_MODE=1 -DREPS=${MAX_REPS} -DEXACT_SVP -DUSE_MIN_EVAP -DUSE_MPIF90"
	FT_WFLAGS = OPT="-DRUN_MODE=2 -DREPS=${MAX_REPS} -DEXACT_SVP -DUSE_MIN_EVAP -DUSE_MPIF90"
	FT_SFLAGS = OPT="-DRUN_MODE=3 -DREPS=${MAX_REPS} -DEXACT_SVP -DUSE_MIN_EVAP -DUSE_MPIF90"
endif

VPROC = 1
ifeq (${IS_DOCKER}, False)
	BPROC = 12
else
	BPROC = 4
endif


.PHONY: all
all: validation benchmark


# Print which version is being used
.PHONY: print
print:
ifeq (${VERSION}, fortran)
	@printf ">> Fortran (original code)"
else ifeq (${VERSION}, python)
	@printf ">> Python (array programming)"
else ifeq (${VERSION}, numpy)
	@printf ">> GT4Py (numpy backend)"
else ifeq (${VERSION}, gtx86)
	@printf ">> GT4Py (gtx86 backend)"
else ifeq (${VERSION}, gtcuda)
	@printf ">> GT4Py (gtcuda backend)"
else
	@printf ">> GT4Py (numpy backend)"
endif


# Run the program on Piz Daint or the Docker image
.PHONY: run
run:
ifeq (${VERSION}, fortran)
  ifeq (${IS_DOCKER}, False)
	@cd ./fortran && srun --account=s1053 --partition=cscsci --nodes=${NODES} --ntasks=${PROC} --ntasks-per-node=${PROC} --ntasks-per-core=${T_PER_CORE} -C gpu ./main.x && cd ..
  else
	@cd ./fortran && mpirun -n ${PROC} ./main.x && cd ..
  endif
else
  ifeq (${IS_DOCKER}, False)
	@srun --account=s1053 --partition=cscsci --nodes=1 -C gpu python3 python/main.py
  else
	@python3 python/main.py
  endif
endif


# Get/generate the correct input and output data
.PHONY: get_data
get_data:
	@./get_data.sh
	@${FT_DFLAGS} $(MAKE) WARN=True -s -C ./fortran
	@$(MAKE) run VERSION="fortran" PROC=1 -s
	@$(MAKE) clean -s -C ./fortran

# Run validation
.PHONY: validation
validation:
	@$(MAKE) print -s
	@printf " --- Validation\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
ifeq (${VERSION}, fortran)
	@${FT_VFLAGS} $(MAKE) WARN=True -s -C ./fortran
	@$(MAKE) run PROC=${VPROC} -s
	@$(MAKE) clean -s -C ./fortran
else
	@$(MAKE) run ${VFLAGS} -s
endif
	@printf "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"

# Run benchmark
.PHONY: benchmark
benchmark:
	@$(MAKE) print -s
	@printf " --- Benchmark\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
	@mkdir -p out
ifeq (${VERSION}, fortran)
	@${FT_BFLAGS} $(MAKE) WARN=False -s -C ./fortran
	@$(MAKE) run PROC=${BPROC} -s
	@$(MAKE) clean -s -C ./fortran
else
	@$(MAKE) run ${BFLAGS} -s
endif


# Clean up
.PHONY: clean
clean:
	@$(MAKE) clean -s -C ./fortran
