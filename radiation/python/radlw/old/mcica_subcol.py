import numpy as np
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from radlw_param import nbands, ngptlw
from radphysparam import iovrlw

def mcica_subcol(cldf, nlay, ipseed, dz, de_lgth):
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    #  input variables:                                                size !
    #   cldf    - real, layer cloud fraction                           nlay !
    #   nlay    - integer, number of model vertical layers               1  !
    #   ipseed  - integer, permute seed for random num generator         1  !
    #    ** note : if the cloud generator is called multiple times, need    !
    #              to permute the seed between each call; if between calls  !
    #              for lw and sw, use values differ by the number of g-pts. !
    #   dz      - real, layer thickness (km)                           nlay !
    #   de_lgth - real, layer cloud decorrelation length (km)            1  !
    #                                                                       !
    #  output variables:                                                    !
    #   lcloudy - logical, sub-colum cloud profile flag array    ngptlw*nlay!
    #                                                                       !
    #  other control flags from module variables:                           !
    #     iovrlw    : control flag for cloud overlapping method             !
    #                 =0:random; =1:maximum/random: =2:maximum; =3:decorr   !
    #                                                                       !
    #  =====================    end of definitions    ====================  !

    lcloudy = np.zeros((ngptlw, nlay))
    cdfunc = np.zeros((ngptlw, nlay))
    rand1d = np.zeros(ngptlw)
    rand2d = np.zeros(nlay*ngptlw)
    fac_lcf = np.zeros(nlay)
    cdfun2 = np.zeros((ngptlw, nlay))

    #
    #===> ...  begin here
    #
    #  --- ...  advance randum number generator by ipseed values

    stat = random_setseed(ipseed)

    #  --- ...  sub-column set up according to overlapping assumption

    if iovrlw == 0:
        # random overlap, pick a random value at every level

        rand2d, stat = random_number()

        k1 = 0
        for n in range(ngptlw):
            for k in range(nlay):
                k1 += 1
                cdfunc[n, k] = rand2d[k1]

    elif iovrlw == 1:    # max-ran overlap

        rand2d, stat = random_number()

        k1 = 0
        for n in range(ngptlw):
            for k in range(nlay):
                k1 += 1
                cdfunc[n, k] = rand2d[k1]

        #  ---  first pick a random number for bottom (or top) layer.
        #       then walk up the column: (aer's code)
        #       if layer below is cloudy, use the same rand num in the layer below
        #       if layer below is clear,  use a new random number

        #  ---  from bottom up
        for k in range(1, nlay):
            k1 = k - 1
            tem1 = 1.0 - cldf[k1]

            for n in range(ngptlw):
                if cdfunc[n, k1] > tem1:
                    cdfunc[n, k] = cdfunc[n, k1]
                else:
                    cdfunc[n, k] = cdfunc[n, k] * tem1

    elif iovrlw == 2:        # maximum overlap, pick same random numebr at every level

        rand1d, stat = random_number()

        for n in range(ngptlw):
            tem1 = rand1d[n]

            for k in range(nlay):
                cdfunc[n, k] = tem1

    elif iovrlw == 3:        # decorrelation length overlap

        #  ---  compute overlapping factors based on layer midpoint distances
        #       and decorrelation depths

        for k in range(nlay-1, 0, -1):
            fac_lcf[k] = np.exp(-0.5 * (dz[k]+dz[k-1]) / de_lgth)

        #  ---  setup 2 sets of random numbers

        rand2d, stat = random_number()

        k1 = 0
        for k in range(nlay):
            for n in range(ngptlw):
                k1 += 1
                cdfunc[n, k] = rand2d[k1]

        rand2d, stat = random_number()

        k1 = 0
        for k in range(nlay):
            for n in range(ngptlw):
                k1 += 1
                cdfun2[n, k] = rand2d[k1]

        #  ---  then working from the top down:
        #       if a random number (from an independent set -cdfun2) is smaller then the
        #       scale factor: use the upper layer's number,  otherwise use a new random
        #       number (keep the original assigned one).

        for k in range(nlay-2, None, -1):
            k1 = k + 1

            for n in range(ngptlw):
                if cdfun2[n, k] <= fac_lcf[k1]:
                    cdfunc[n, k] = cdfunc[n, k1]

    #  --- ...  generate subcolumns for homogeneous clouds

    for k in range(nlay):
        tem1 = 1.0 - cldf[k]

        for n in range(ngptlw):
            lcloudy[n, k] = cdfunc[n, k] >= tem1

    return lcloudy