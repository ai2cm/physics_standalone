import numpy as np

from config import *
import serialbox as ser
from radiation_driver import RadiationDriver

serializer = ser.Serializer(
    ser.OpenModeKind.Read, "/work/radiation/fortran/data/LW", "Generator_rank0"
)

serializer2 = ser.Serializer(
    ser.OpenModeKind.Read, "/work/radiation/fortran/data/SW", "Generator_rank0"
)

model_vars = [
    "me",
    "levr",
    "levs",
    "nfxr",
    "ntrac",
    "ntcw",
    "ntiw",
    "ncld",
    "ntrw",
    "ntsw",
    "ntgl",
    "ncnd",
    "fhswr",
    "fhlwr",
    "ntoz",
    "lsswr",
    "solhr",
    "lslwr",
    "imp_physics",
    "lgfdlmprad",
    "uni_cld",
    "effr_in",
    "indcld",
    "ntclamt",
    "num_p3d",
    "npdf3d",
    "ncnvcld3d",
    "lmfdeep2",
    "sup",
    "kdt",
    "lmfshal",
    "do_sfcperts",
    "pertalb",
    "do_only_clearsky_rad",
    "swhtr",
    "solcon",
    "lprnt",
    "lwhtr",
]

Model = dict()
for var in model_vars:
    Model[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

statein_vars = [
    "prsi",
    "prsl",
    "tgrs",
    "prslk",
    "qgrs",
]

Statein = dict()
for var in statein_vars:
    Statein[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

sfcprop_vars = [
    "tsfc",
    "slmsk",
    "snowd",
    "sncovr",
    "snoalb",
    "zorl",
    "hprime",
    "alvsf",
    "alnsf",
    "alvwf",
    "alnwf",
    "facsf",
    "facwf",
    "fice",
    "tisfc",
]

Sfcprop = dict()
for var in sfcprop_vars:
    Sfcprop[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

coupling_vars = [
    "nirbmdi",
    "nirdfdi",
    "visbmdi",
    "visdfdi",
    "nirbmui",
    "nirdfui",
    "visbmui",
    "visdfui",
    "sfcnsw",
    "sfcdsw",
    "sfcdlw",
]

Coupling = dict()
for var in coupling_vars:
    Coupling[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

grid_vars = [
    "xlon",
    "xlat",
    "sinlat",
    "coslat",
]

Grid = dict()
for var in grid_vars:
    Grid[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

tbd_vars = [
    "phy_f3d",
    "icsdsw",
    "icsdlw",
]

Tbd = dict()
for var in tbd_vars:
    Tbd[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

radtend_vars = [
    "coszen",
    "coszdg",
    "sfalb",
    "htrsw",
    "swhc",
    "lwhc",
    "semis",
    "tsflw",
]

Radtend = dict()
for var in radtend_vars:
    Radtend[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

Radtend["sfcfsw"] = dict()
Radtend["sfcflw"] = dict()

diag_vars = ["fluxr"]
Diag = dict()
for var in diag_vars:
    Diag[var] = serializer.read(var, serializer.savepoint["driver-in-000000"])

Diag["topflw"] = dict()
Diag["topfsw"] = dict()


def getscalars(indict):
    for var in indict.keys():
        if not type(indict[var]) == dict:
            if indict[var].size == 1:
                indict[var] = indict[var][0]

    return indict


si = serializer2.read("si", serializer2.savepoint["rad-initialize"])
imp_physics = serializer2.read("imp_physics", serializer2.savepoint["rad-initialize"])


Model = getscalars(Model)
Statein = getscalars(Statein)
Sfcprop = getscalars(Sfcprop)
Coupling = getscalars(Coupling)
Grid = getscalars(Grid)
Tbd = getscalars(Tbd)
Radtend = getscalars(Radtend)
Diag = getscalars(Diag)

isolar = 2  # solar constant control flag
ictmflg = 1  # data ic time/date control flag
ico2flg = 2  # co2 data source control flag
ioznflg = 7  # ozone data source control flag

iaer = 111

if ictmflg == 0 or ictmflg == -2:
    iaerflg = iaer % 100  # no volcanic aerosols for clim hindcast
else:
    iaerflg = iaer % 1000

iaermdl = iaer / 1000  # control flag for aerosol scheme selection
if iaermdl < 0 or iaermdl > 2 and iaermdl != 5:
    print("Error -- IAER flag is incorrect, Abort")

iswcliq = 1  # optical property for liquid clouds for sw
iovrsw = 1  # cloud overlapping control flag for sw
iovrlw = 1  # cloud overlapping control flag for lw
lcrick = False  # control flag for eliminating CRICK
lcnorm = False  # control flag for in-cld condensate
lnoprec = False  # precip effect on radiation flag (ferrier microphysics)
isubcsw = 2  # sub-column cloud approx flag in sw radiation
isubclw = 2  # sub-column cloud approx flag in lw radiation
ialbflg = 1  # surface albedo control flag
iemsflg = 1  # surface emissivity control flag
icldflg = 1
ivflip = 1  # vertical index direction control flag

driver = RadiationDriver()
driver.radinit(
    si,
    nlay,
    imp_physics,
    Model["me"],
    iemsflg,
    ioznflg,
    ictmflg,
    isolar,
    ico2flg,
    iaerflg,
    ialbflg,
    icldflg,
    ivflip,
    iovrsw,
    iovrlw,
    isubcsw,
    isubclw,
    lcrick,
    lcnorm,
    lnoprec,
    iswcliq,
)

Radtendout, Diagout = driver.GFS_radiation_driver(
    Model, Statein, Sfcprop, Coupling, Grid, Tbd, Radtend, Diag
)
