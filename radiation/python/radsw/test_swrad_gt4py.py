import numpy as np
import sys

sys.path.insert(0, "/deployed/radiation/python")
from radsw.radsw_main_gt4py import RadSWClass
from radphysparam import icldflg

me = 0
iovrsw = 1
isubcsw = 2

for rank in range(1):
    rsw = RadSWClass(rank, iovrsw, isubcsw, icldflg)
    rsw.create_input_data(rank)
    rsw.lwrad(rank, do_subtest=True)
