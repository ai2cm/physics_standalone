import numpy as np
import sys

sys.path.insert(0, "..")
from radlw.radlw_main_gt4py import RadLWClass

me = 0
iovrlw = 1
isubclw = 2

rlw = RadLWClass(me, iovrlw, isubclw)

for tile in range(6):
    rlw.create_input_data(tile)
    rlw.lwrad(do_subtest=True)
