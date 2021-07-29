import os
import numpy as np
import xarray as xr
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import nbands, nrates, delwave


def setcoef(pavel, tavel, tz, stemp, h2ovmr, colamt, coldry, colbrd,
            nlay, nlp1):

    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  inputs:                                                       -size- !
    #   pavel     - real, layer pressures (mb)                         nlay !
    #   tavel     - real, layer temperatures (k)                       nlay !
    #   tz        - real, level (interface) temperatures (k)         0:nlay !
    #   stemp     - real, surface ground temperature (k)                1   !
    #   h2ovmr    - real, layer w.v. volum mixing ratio (kg/kg)        nlay !
    #   colamt    - real, column amounts of absorbing gases      nlay*maxgas!
    #                 2nd indices range: 1-maxgas, for watervapor,          !
    #                 carbon dioxide, ozone, nitrous oxide, methane,        !
    #                 oxigen, carbon monoxide,etc. (molecules/cm**2)        !
    #   coldry    - real, dry air column amount                        nlay !
    #   colbrd    - real, column amount of broadening gases            nlay !
    #   nlay/nlp1 - integer, total number of vertical layers, levels    1   !
    #                                                                       !
    #  outputs:                                                             !
    #   laytrop   - integer, tropopause layer index (unitless)          1   !
    #   pklay     - real, integrated planck func at lay temp   nbands*0:nlay!
    #   pklev     - real, integrated planck func at lev temp   nbands*0:nlay!
    #   jp        - real, indices of lower reference pressure          nlay !
    #   jt, jt1   - real, indices of lower reference temperatures      nlay !
    #   rfrate    - real, ref ratios of binary species param   nlay*nrates*2!
    #     (:,m,:)m=1-h2o/co2,2-h2o/o3,3-h2o/n2o,4-h2o/ch4,5-n2o/co2,6-o3/co2!
    #     (:,:,n)n=1,2: the rates of ref press at the 2 sides of the layer  !
    #   facij     - real, factors multiply the reference ks,           nlay !
    #                 i,j=0/1 for lower/higher of the 2 appropriate         !
    #                 temperatures and altitudes.                           !
    #   selffac   - real, scale factor for w. v. self-continuum        nlay !
    #                 equals (w. v. density)/(atmospheric density           !
    #                 at 296k and 1013 mb)                                  !
    #   selffrac  - real, factor for temperature interpolation of      nlay !
    #                 reference w. v. self-continuum data                   !
    #   indself   - integer, index of lower ref temp for selffac       nlay !
    #   forfac    - real, scale factor for w. v. foreign-continuum     nlay !
    #   forfrac   - real, factor for temperature interpolation of      nlay !
    #                 reference w.v. foreign-continuum data                 !
    #   indfor    - integer, index of lower ref temp for forfac        nlay !
    #   minorfrac - real, factor for minor gases                       nlay !
    #   scaleminor,scaleminorn2                                             !
    #             - real, scale factors for minor gases                nlay !
    #   indminor  - integer, index of lower ref temp for minor gases   nlay !
    #                                                                       !
    #  ======================    end of definitions    ===================  !


    #===> ... begin here
    #
    #  --- ...  calculate information needed by the radiative transfer routine
    #           that is specific to this atmosphere, especially some of the
    #           coefficients and indices needed to compute the optical depths
    #           by interpolating data from stored reference atmospheres.

    dfile = '../lookupdata/totplnk.nc'
    pfile = '../lookupdata/radlw_ref_data.nc'
    totplnk = xr.open_dataset(dfile)['totplnk'].data
    preflog = xr.open_dataset(pfile)['preflog'].data
    tref = xr.open_dataset(pfile)['tref'].data
    chi_mls = xr.open_dataset(pfile)['chi_mls'].data

    pklay = np.zeros((nbands, nlp1))
    pklev = np.zeros((nbands, nlp1))

    jp = np.zeros(nlay, dtype=np.int32)
    jt = np.zeros(nlay, dtype=np.int32)
    jt1 = np.zeros(nlay, dtype=np.int32)
    fac00 = np.zeros(nlay)
    fac01 = np.zeros(nlay)
    fac10 = np.zeros(nlay)
    fac11 = np.zeros(nlay)
    forfac = np.zeros(nlay)
    forfrac = np.zeros(nlay)
    selffac = np.zeros(nlay)
    scaleminor = np.zeros(nlay)
    scaleminorn2 = np.zeros(nlay)
    indminor = np.zeros(nlay)
    minorfrac = np.zeros(nlay)
    indfor = np.zeros(nlay)
    indself = np.zeros(nlay)
    selffrac = np.zeros(nlay)
    rfrate = np.zeros((nlay, nrates, 2))

    f_zero = 0.0
    f_one = 1.0
    stpfac  = 296.0/1013.0


    indlay = np.minimum(180, np.maximum(1, int(stemp-159.0)))
    indlev = np.minimum(180, np.maximum(1, int(tz[0]-159.0) ))
    tlyrfr = stemp - int(stemp)
    tlvlfr = tz[0] - int(tz[0])

    for i in range(nbands):
        tem1 = totplnk[indlay, i] - totplnk[indlay-1, i]
        tem2 = totplnk[indlev, i] - totplnk[indlev-1, i]
        pklay[i, 0] = delwave[i] * (totplnk[indlay-1, i] + tlyrfr*tem1)
        pklev[i, 0] = delwave[i] * (totplnk[indlev-1, i] + tlvlfr*tem2)


    #  --- ...  begin layer loop
    #           calculate the integrated Planck functions for each band at the
    #           surface, level, and layer temperatures.

    laytrop = 0

    for k in range(nlay):
        indlay = np.minimum(180, np.maximum(1, int(tavel[k]-159.0)))
        tlyrfr = tavel[k] - int(tavel[k])

        indlev = np.minimum(180, np.maximum(1, int(tz[k+1]-159.0)))
        tlvlfr = tz[k+1] - int(tz[k+1])

        #  --- ...  begin spectral band loop

        for i in range(nbands):
            pklay[i, k+1] = delwave[i] * (totplnk[indlay-1, i] + tlyrfr
                                        * (totplnk[indlay, i] - \
                totplnk[indlay-1, i]))
            pklev[i, k+1] = delwave[i] * (totplnk[indlev-1, i] + tlvlfr
                                        * (totplnk[indlev, i] - \
                totplnk[indlev-1, i]))

        #  --- ...  find the two reference pressures on either side of the
        #           layer pressure. store them in jp and jp1. store in fp the
        #           fraction of the difference (in ln(pressure)) between these
        #           two values that the layer pressure lies.

        plog = np.log(pavel[k])
        jp[k] = np.maximum(1, np.minimum(58, int(36.0 - 5.0*(plog+0.04))))-1
        jp1 = jp[k] + 1
        #  --- ...  limit pressure extrapolation at the top
        fp = np.maximum(f_zero, np.minimum(f_one, 5.0*(preflog[jp[k]]-plog)))

        #  --- ...  determine, for each reference pressure (jp and jp1), which
        #           reference temperature (these are different for each
        #           reference pressure) is nearest the layer temperature but does
        #           not exceed it. store these indices in jt and jt1, resp.
        #           store in ft (resp. ft1) the fraction of the way between jt
        #           (jt1) and the next highest reference temperature that the
        #           layer temperature falls.

        tem1 = (tavel[k]-tref[jp[k]]) / 15.0
        tem2 = (tavel[k]-tref[jp1]) / 15.0
        jt[k] = np.maximum(1, np.minimum(4, int(3.0 + tem1)))-1
        jt1[k] = np.maximum(1, np.minimum(4, int(3.0 + tem2)))-1
        #  --- ...  restrict extrapolation ranges by limiting abs(det t) < 37.5 deg
        ft  = np.maximum(-0.5, np.minimum(1.5, tem1 - float(jt[k] - 2)))
        ft1 = np.maximum(-0.5, np.minimum(1.5, tem2 - float(jt1[k] - 2)))

        #  --- ...  we have now isolated the layer ln pressure and temperature,
        #           between two reference pressures and two reference temperatures
        #           (for each reference pressure).  we multiply the pressure
        #           fraction fp with the appropriate temperature fractions to get
        #           the factors that will be needed for the interpolation that yields
        #           the optical depths (performed in routines taugbn for band n)

        tem1 = f_one - fp
        fac10[k] = tem1 * ft
        fac00[k] = tem1 * (f_one - ft)
        fac11[k] = fp * ft1
        fac01[k] = fp * (f_one - ft1)

        forfac[k] = pavel[k]*stpfac / (tavel[k]*(1.0 + h2ovmr[k]))
        selffac[k] = h2ovmr[k] * forfac[k]

        #  --- ...  set up factors needed to separately include the minor gases
        #           in the calculation of absorption coefficient

        scaleminor[k] = pavel[k] / tavel[k]
        scaleminorn2[k] = (pavel[k] / tavel[k]) * \
            (colbrd[k]/(coldry[k] + colamt[k, 0]))
        tem1 = (tavel[k] - 180.8) / 7.2
        indminor[k] = np.minimum(18, np.maximum(1, int(tem1)))
        minorfrac[k] = tem1 - float(indminor[k])

        #  --- ...  if the pressure is less than ~100mb, perform a different
        #           set of species interpolations.

        if plog > 4.56:
            laytrop =  laytrop + 1

            tem1 = (332.0 - tavel[k]) / 36.0
            indfor[k] = np.minimum(2, np.maximum(1, int(tem1)))
            forfrac[k] = tem1 - float(indfor[k])

            #  --- ...  set up factors needed to separately include the water vapor
            #           self-continuum in the calculation of absorption coefficient.

            tem1 = (tavel[k] - 188.0) / 7.2
            indself[k] = np.minimum(9, np.maximum(1, int(tem1)-7))
            selffrac[k] = tem1 - float(indself[k] + 7)

            #  --- ...  setup reference ratio to be used in calculation of binary
            #           species parameter in lower atmosphere.

            rfrate[k, 0, 0] = chi_mls[0, jp[k]] / chi_mls[1, jp[k]]
            rfrate[k, 0, 1] = chi_mls[0, jp[k]+1] / chi_mls[1, jp[k]+1]

            rfrate[k, 1, 0] = chi_mls[0, jp[k]] / chi_mls[2, jp[k]]
            rfrate[k, 1, 1] = chi_mls[0, jp[k]+1] / chi_mls[2, jp[k]+1]

            rfrate[k, 2, 0] = chi_mls[0, jp[k]] / chi_mls[3, jp[k]]
            rfrate[k, 2, 1] = chi_mls[0, jp[k]+1] / chi_mls[3, jp[k]+1]

            rfrate[k, 3, 0] = chi_mls[0, jp[k]] / chi_mls[5, jp[k]]
            rfrate[k, 3, 1] = chi_mls[0, jp[k]+1] / chi_mls[5, jp[k]+1]

            rfrate[k, 4, 0] = chi_mls[3, jp[k]] / chi_mls[1, jp[k]]
            rfrate[k, 4, 1] = chi_mls[3, jp[k]+1] / chi_mls[1, jp[k]+1]

        else:

            tem1 = (tavel[k] - 188.0) / 36.0
            indfor[k] = 3
            forfrac[k] = tem1 - f_one

            indself[k] = 0
            selffrac[k] = f_zero

            #  --- ...  setup reference ratio to be used in calculation of binary
            #           species parameter in upper atmosphere.

            rfrate[k, 0, 0] = chi_mls[0, jp[k]] / chi_mls[1, jp[k]]
            rfrate[k, 0, 1] = chi_mls[0, jp[k]+1] / chi_mls[1, jp[k]+1]

            rfrate[k, 5, 0] = chi_mls[2, jp[k]] / chi_mls[1, jp[k]]
            rfrate[k, 5, 1] = chi_mls[2, jp[k]+1] / chi_mls[1, jp[k]+1]

        #  --- ...  rescale selffac and forfac for use in taumol

        selffac[k] = colamt[k, 0] * selffac[k]
        forfac[k]  = colamt[k, 0] * forfac[k]

    return (laytrop, pklay, pklev, jp, jt, jt1, rfrate, fac00, fac01, fac10,
            fac11, selffac, selffrac, indself, forfac, forfrac, indfor,
            minorfrac, scaleminor, scaleminorn2, indminor)

os.environ["DYLD_LIBRARY_PATH"]="/Users/AndrewP/Documents/code/serialbox2/install/lib"

SERIALBOX_DIR = "/Users/AndrewP/Documents/code/serialbox2/install"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

ddir = '/Users/AndrewP/Documents/work/physics_standalone/radiation/fortran/radlw/dump'

serializer = ser.Serializer(ser.OpenModeKind.Read, ddir, "Serialized_rank0")
savepoints = serializer.savepoint_list()
print(savepoints[10])

# print(savepoints)

invars = ['pavel', 'tavel', 'tz', 'stemp', 'h2ovmr', 'colamt',
          'coldry', 'colbrd', 'nlay', 'nlp1']

outvars = ['laytrop', 'pklay', 'pklev', 'jp', 'jt', 'jt1',
            'rfrate', 'fac00', 'fac01', 'fac10', 'fac11',
            'selffac', 'selffrac', 'indself', 'forfac', 'forfrac', 'indfor',
            'minorfrac', 'scaleminor', 'scaleminorn2', 'indminor']

indict = dict()
outdict = dict()

for var in invars:
    tmp = serializer.read(var, savepoints[10])

    indict[var] = tmp

for var in outvars:
    tmp = serializer.read(var, savepoints[11])

    outdict[var] = tmp

test = setcoef(indict['pavel'],
               indict['tavel'],
               indict['tz'],
               indict['stemp'],
               indict['h2ovmr'],
               indict['colamt'],
               indict['coldry'],
               indict['colbrd'],
               indict['nlay'][0],
               indict['nlp1'][0])

valdict = dict()
for n, var in enumerate(outvars):
    valdict[var] = test[n]

def compare_data(data, ref_data, explicit=True, blocking=True):

    wrong = []
    flag = True

    for var in data:

        # Fix indexing for fortran vs python
        if var == 'jp' or var == 'jt' or var == 'jt1':
            if not np.allclose(
                data[var]+1, ref_data[var], rtol=1e-11, atol=1.0e-13, equal_nan=True
            ):

                wrong.append(var)
                flag = False

            else:

                if explicit:
                    print(f"Successfully validated {var}!")
        else:
            if not np.allclose(
                data[var], ref_data[var], rtol=1e-11, atol=1.0e-13, equal_nan=True
            ):

                wrong.append(var)
                flag = False

            else:

                if explicit:
                    print(f"Successfully validated {var}!")

    if blocking:
        assert flag, f"Output data does not match reference data for field {wrong}!"
    else:
        if not flag:
            print(f"Output data does not match reference data for field {wrong}!")

compare_data(valdict, outdict)
