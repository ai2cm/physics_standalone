
SERIALBOX_DIR ?= /usr/local/serialbox
NETCDF_LIB ?= /lib/x86_64-linux-gnu
NCEPLIBS_DIR ?= /NCEPlibs
NCEPLIBS = -L$(NCEPLIBS_DIR) -lnemsio_d -lbacio_4 -lsp_v2.0.2_d -lw3emc_d -lw3nco_d

TARGET = main.x

SRC = \
  m_countdown.F \
  machine.F \
  iounitdef.F \
  physparam.F \
  physcons.F \
  funcphys.F \
  ozne_def.F \
  h2o_def.F \
  aerclm_def.F \
  rascnvv2.F \
  wam_f107_kp_mod.F \
  gocart_tracer_config_stub.F \
  GFDL_parse_tracers.F \
  gfs_phy_tracer_config.F \
  radsw_param.F \
  radlw_param.F \
  radlw_datatb.F  \
  radsw_datatb.F \
  mersenne_twister.F \
  module_bfmicrophysics.F \
  surface_perturbation.F \
  radiation_astronomy.F \
  radiation_surface.F \
  radiation_gases.F \
  radiation_clouds.F \
  radiation_aerosols.F \
  radsw_main2.F \
  radlw_main.F  \
  GFS_typedefs.F \
  GFS_radiation_driver.F \
  rad_initialize.F \
  main.F

OBJ = $(SRC:.F=.o)
PP = $(SRC:.F=.F90)

F90 = mpif90
#FFLAGS = -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none \
#  -fno-backslash -fimplicit-none -fno-range-check -pedantic -Waliasing -Wampersand \
#  -Wline-truncation -Wsurprising -Wtabs -Wunderflow -O0 -g -fbacktrace \
#  -fdump-core -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan \
#  -finit-integer=9999999 -finit-logical=true -finit-character=35 \
#  -DSERIALIZE -I$(SERIALBOX_DIR)/include
FFLAGS = -cpp -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none -finit-local-zero\
  -fno-backslash -fimplicit-none -fno-range-check -O3 -ftree-vectorize -funroll-loops \
  -DSERIALIZE -I$(SERIALBOX_DIR)/include

LD = mpif90
LDFLAGS = -O3 -ftree-vectorize -funroll-loops $(SERIALBOX_DIR)/lib/libSerialboxFortran.a $(SERIALBOX_DIR)/lib/libSerialboxC.a \
  $(SERIALBOX_DIR)/lib/libSerialboxCore.a -L$(NETCDF_LIB) -lnetcdff -lnetcdf \
  -lpthread -lstdc++ -lstdc++fs

.PHONY: all
all: pre-process $(TARGET)

.PHONY: pre-process
pre-process: $(PP)

.PHONY: clean
clean:
	/bin/rm -rf *.mod *.o $(PP) __pycache__ $(TARGET) dump/*

.PHONY: distclean
distclean: clean
	/bin/rm -rf dump

$(TARGET): $(OBJ)
	$(LD) $(OBJ) $(LDFLAGS) $(NCEPLIBS) -o $(TARGET)

%.F90: %.F
	$(SERIALBOX_DIR)/python/pp_ser/pp_ser.py --sp-as-var --no-prefix -v --output=$@ $<

%.o: %.F90
	$(F90) $(FFLAGS) -c $< -o $@

