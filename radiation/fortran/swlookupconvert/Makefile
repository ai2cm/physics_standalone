SERIALBOX_DIR ?= /usr/local/serialbox
NETCDF_LIB ?= /lib/x86_64-linux-gnu

TARGET = main.x

SRC = \
  machine.F \
  physparam.F \
  radsw_param.F \
  radsw_datatb.F \
  serialize_datatb.F

OBJ = $(SRC:.F=.o)
PP = $(SRC:.F=.F90)

F90 = mpif90
# FFLAGS = -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none \
#   -fno-backslash -fimplicit-none -frange-check -pedantic -Waliasing -Wampersand \
#   -Wline-truncation -Wsurprising -Wtabs -Wunderflow -O0 -g -fbacktrace \
#   -fdump-core -ffpe-trap=invalid,zero,overflow -fbounds-check -finit-real=nan \
#   -finit-integer=9999999 -finit-logical=true -finit-character=35 \
#   -DSERIALIZE -I$(SERIALBOX_DIR)/include
# FFLAGS = -fdec -fdefault-real-8 -fno-fast-math -ffree-form -ffree-line-length-none \
#   -fno-backslash -fimplicit-none -pedantic -Waliasing -Wampersand \
#   -Wline-truncation -Wsurprising -Wtabs -Wunderflow -O3 -ftree-vectorize -funroll-loops \
#   -DSERIALIZE -I$(SERIALBOX_DIR)/include

FFLAGS = -fdec -fdefault-real-8 -O3 -DSERIALIZE -I$(SERIALBOX_DIR)/include

LD = mpif90
LDFLAGS = $(SERIALBOX_DIR)/lib/libSerialboxFortran.a $(SERIALBOX_DIR)/lib/libSerialboxC.a \
  $(SERIALBOX_DIR)/lib/libSerialboxCore.a -L$(NETCDF_LIB) -lnetcdff -lnetcdf\
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
	$(LD) $(OBJ) $(LDFLAGS) -o $(TARGET)

%.F90: %.F
	$(SERIALBOX_DIR)/python/pp_ser/pp_ser.py --sp-as-var --no-prefix -v --output=$@ $<

%.o: %.F90
	$(F90) $(FFLAGS) -c $< -o $@

