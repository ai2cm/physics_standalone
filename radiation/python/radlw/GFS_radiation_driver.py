import os
import sys
import numpy as np
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')

VTAGRAD = 'NCEP-Radiation_driver    v5.2  Jan 2013'

QMIN=1.0e-10
QME5=1.0e-7
QME6=1.0e-7
EPSQ=1.0e-12
prsmin = 1.0e-6
itsfc = 0
month0 = 0
iyear0 = 0
monthd = 0
loz1st = True
LTP = 0
lextop = LTP > 0

def radinit(si, NLAY, imp_physics, me):
    itsfc  = iemsflg / 10             # sfc air/ground temp control
    loz1st = ioznflg == 0           # first-time clim ozone data read flag
    month0 = 0
    iyear0 = 0
    monthd = 0

