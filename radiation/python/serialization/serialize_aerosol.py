import sys
import numpy as np
import os
import xarray as xr

# On MacOS, remember to set the environment variable DYLD_LIBRARY_PATH to contain
# the path to the SerialBox /lib directory

os.environ["DYLD_LIBRARY_PATH"] = "/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = "/Users/andrewp/Documents/work/physics_standalone/radiation/fortran/radlw/dump"

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Init_rank0")
savepoints = serializer.savepoint_list()

sp = serializer.savepoint["lwrad-clim_aerinit-output"]
sp2 = serializer.savepoint["lw_aer_update_out000000"]

iendwv = serializer.read("iendwv", sp)
haer = serializer.read("haer", sp)
prsref = serializer.read("prsref", sp)
rhidext0 = serializer.read("rhidext0", sp)
rhidsca0 = serializer.read("rhidsca0", sp)
rhidssa0 = serializer.read("rhidssa0", sp)
rhidasy0 = serializer.read("rhidasy0", sp)
rhdpext0 = serializer.read("rhdpext0", sp)
rhdpsca0 = serializer.read("rhdpsca0", sp)
rhdpssa0 = serializer.read("rhdpssa0", sp)
rhdpasy0 = serializer.read("rhdpasy0", sp)
straext0 = serializer.read("straext0", sp)

kprfg = serializer.read("kprfg", sp2)
idxcg = serializer.read("idxcg", sp2)
cmixg = serializer.read("cmixg", sp2)
denng = serializer.read("denng", sp2)

print(rhdpasy0.shape)

NCM1 = 6
NSWLWBD = 30
NRHLEV = 8
NCM2 = 4
NBANDS = 61
NDOMAINS = 5
NPROFS = 7
IMXAE = 72
JMXAE = 37
NXC = 5
Z = 2

cline = [
    "MONTH OF JANUARY   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF FEBRUARY   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF MARCH   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF APRIL   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF MAY   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF JUNE   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF JULY   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF AUGUST   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF SEPTEMBER   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF OCTOBER   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF NOVEMBER   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
    "MONTH OF DECEMBER   CLIMATOLOGICAL AEROSOL GLOBAL DISTRIBUTION",
]

ds = xr.Dataset(
    {
        "iendwv": ("NBANDS", iendwv),
        "haer": (("NDOMAINS", "NPROFS"), haer),
        "prsref": (("NDOMAINS", "NPROFS"), prsref),
        "rhidext0": (("NBANDS", "NCM1"), rhidext0),
        "rhidsca0": (("NBANDS", "NCM1"), rhidsca0),
        "rhidssa0": (("NBANDS", "NCM1"), rhidssa0),
        "rhidasy0": (("NBANDS", "NCM1"), rhidasy0),
        "rhdpext0": (("NBANDS", "NRHLEV", "NCM2"), rhdpext0),
        "rhdpsca0": (("NBANDS", "NRHLEV", "NCM2"), rhdpsca0),
        "rhdpssa0": (("NBANDS", "NRHLEV", "NCM2"), rhdpssa0),
        "rhdpasy0": (("NBANDS", "NRHLEV", "NCM2"), rhdpasy0),
        "straext0": ("NBANDS", straext0),
        "kprfg": (("IMXAE", "JMXAE"), kprfg),
        "cmixg": (("NXC", "IMXAE", "JMXAE"), cmixg),
        "idxcg": (("NXC", "IMXAE", "JMXAE"), idxcg),
        "denng": (("Z", "IMXAE", "JMXAE"), denng),
        "cline": ("month", cline),
    }
)

print(ds)

dout = (
    "/Users/andrewp/Documents/work/physics_standalone/radiation/python/radlw/aerosol.nc"
)

ds.to_netcdf(dout)
