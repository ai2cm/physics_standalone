import sys
import os
import numpy as np
import xarray as xr
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radphysparam import semis_file

class SurfaceClass():
    VTAGSFC = 'NCEP-Radiation_surface   v5.1  Nov 2012'
    IMXEMS = 360
    JMXEMS = 180

    def __init__(self, me, ialb, iems):

        self.ialbflg = ialb
        self.iemsflg = iems

        if me == 0:
            print(self.VTAGSFC)   # print out version tag

        # - Initialization of surface albedo section
        # physparam::ialbflg 
        # = 0: using climatology surface albedo scheme for SW
        # = 1: using MODIS based land surface albedo for SW

        if self.ialbflg == 0:
            if me == 0:
                print('- Using climatology surface albedo scheme for sw')

        elif self.ialbflg == 1:
            if me == 0:
                print('- Using MODIS based land surface albedo for sw')
        else:
            print(f'!! ERROR in Albedo Scheme Setting, IALB={self.ialbflg}')

        # - Initialization of surface emissivity section
        # physparam::iemsflg
        # = 0: fixed SFC emissivity at 1.0
        # = 1: input SFC emissivity type map from "semis_file"

        iemslw = self.iemsflg % 10          # emissivity control
        if iemslw == 0:                # fixed sfc emis at 1.0
            if me == 0:
                print('- Using Fixed Surface Emissivity = 1.0 for lw')

        elif iemslw == 1:              # input sfc emiss type map
            if 'idxems' not in vars():
                idxems = np.zeros((self.IMXEMS, self.JMXEMS))

                file_exist = os.path.isfile(semis_file)

                if not file_exist:
                    if me == 0:
                        print('- Using Varying Surface Emissivity for lw')
                        print(f'Requested data file "{semis_file}" not found!')
                        print('Change to fixed surface emissivity = 1.0 !')

                    iemslw = 0
                else:
                    ds = xr.open_dataset(semis_file)

                    cline = ds['cline'].data
                    idxems = ds['idxems'].data

                    if me == 0:
                        print('- Using Varying Surface Emissivity for lw')
                        print(f'Opened data file: {semis_file}')
                        print(cline)
            else:
                print(f'!! ERROR in Emissivity Scheme Setting, IEMS={self.iemsflg}')

        
        self.cline = cline
        self.idxems = idxems

    def return_initdata(self):
        outdict = {'idxems': self.idxems}
        return outdict