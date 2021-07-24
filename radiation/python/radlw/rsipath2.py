import numpy as np
import os
import sys
sys.path.insert(0, '/Users/AndrewP/Documents/work/physics_standalone/radiation/python')
from phys_const import con_fvirt, con_rd, con_g, con_t0c, con_pi

def rsipath2(plyr, plvl, tlyr, qlyr, qcwat, qcice, qrain, rrime,
             IM, LEVS, iflip, flgmin):
    # =================   subprogram documentation block   ================ !
    #                                                                       !
    # abstract:  this program is a modified version of ferrier's original   !
    #   "rsipath" subprogram.  it computes layer's cloud liquid, ice, rain, !
    #   and snow water condensate path and the partical effective radius    !
    #   for liquid droplet, rain drop, and snow flake.                      !
    #                                                                       !
    #  ====================  defination of variables  ====================  !
    #                                                                       !
    # input variables:                                                      !
    #   plyr  (IM,LEVS) : model layer mean pressure in mb (100Pa)           !
    #   plvl  (IM,LEVS+1):model level pressure in mb (100Pa)                !
    #   tlyr  (IM,LEVS) : model layer mean temperature in k                 !
    #   qlyr  (IM,LEVS) : layer specific humidity in gm/gm                  !
    #   qcwat (IM,LEVS) : layer cloud liquid water condensate amount        !
    #   qcice (IM,LEVS) : layer cloud ice water condensate amount           !
    #   qrain (IM,LEVS) : layer rain drop water amount                      !
    #   rrime (IM,LEVS) : mass ratio of total to unrimed ice ( >= 1 )       !
    #   IM              : horizontal dimention                              !
    #   LEVS            : vertical layer dimensions                         !
    #   iflip           : control flag for in/out vertical indexing         !
    #                     =0: index from toa to surface                     !
    #                     =1: index from surface to toa                     !
    #   flgmin          : Minimum large ice fraction                        !
    #   lprnt           : logical check print control flag                  !
    #                                                                       !
    # output variables:                                                     !
    #   cwatp (IM,LEVS) : layer cloud liquid water path                     !
    #   cicep (IM,LEVS) : layer cloud ice water path                        !
    #   rainp (IM,LEVS) : layer rain water path                             !
    #   snowp (IM,LEVS) : layer snow water path                             !
    #   recwat(IM,LEVS) : layer cloud eff radius for liqid water (micron)   !
    #   rerain(IM,LEVS) : layer rain water effective radius      (micron)   !
    #   resnow(IM,LEVS) : layer snow flake effective radius      (micron)   !
    #   snden (IM,LEVS) : 1/snow density                                    !
    #                                                                       !
    #                                                                       !
    # usage:     call rsipath2                                              !
    #                                                                       !
    # subroutines called:  none                                             !
    #                                                                       !
    # program history log:                                                  !
    #      xx-xx-2001   b. ferrier     - original program                   !
    #      xx-xx-2004   s. moorthi     - modified for use in gfs model      !
    #      05-20-2004   y. hou         - modified, added vertical index flag!
    #                     to reduce data flipping, and rearrange code to    !
    #                     be comformable with radiation part programs.      !
    #                                                                       !
    #  ====================    end of description    =====================  !
    #
    
    CEXP= 1.0/3.0
    TNW = 50.
    EPSQ = 1.0e-20
    RHOL = 1000.0
    N0r0 = 8.0e6
    FLG1P0 = 1.0
    EPS1 = con_fvirt
    RD = con_rd
    GRAV = con_g
    T0C = con_t0c
    PI = con_pi

    C_N0r0 = PI*RHOL*N0r0
    CN0r0=1.E6/C_N0r0**.25

    recw1 = 620.3505 / TNW**CEXP         # cloud droplet effective radius

    NLImin = 100.

    DMImin = 0.05e-3
    XMImin = 1.0e6*DMImin
    INDEXSmin = 100

    RECImin=1.5*XMImin
    RESNOWmin=1.5*INDEXSmin
    RECWmin=10.

    DMRmin = 0.05e-3
    XMRmin = 1.e6*DMRmin
    RERAINmin = 1.5*XMRmin

    DMRmax=.45e-3
    XMRmax=1.e6*DMRmax

    DMImax=1.e-3
    DMImin=.05e-3
    XMImax=1.e6*DMImax
    XMImin=1.e6*DMImin
    MDImax=XMImax
    MDImin=XMImin

    for i in range(MDImin-1, MDImax):
        SDENS[i] = PI*1.0e-15*float(i*i*i)/MASSI[i]

    # --- hydrometeor's optical path
    cwatp = np.zeros((IM, LEVS))
    cicep = np.zeros((IM, LEVS))
    rainp = np.zeros((IM, LEVS))
    snowp = np.zeros((IM, LEVS))
    snden = np.zeros((IM, LEVS))

    recwat = np.zeros((IM, LEVS))
    rerain = np.zeros((IM, LEVS))
    resnow = np.zeros((IM, LEVS))
    
    for k in range(LEVS):
        for i in range(IM):
            # hydrometeor's effective radius
            recwat[i, k] = RECWmin
            rerain[i, k] = RERAINmin
            resnow[i, k] = RESNOWmin
    
            # set up pressure related arrays, convert unit from mb to cb (10Pa)
            # cause the rest part uses cb in computation
    
            if iflip == 0:        # data from toa to sfc
                ksfc = LEVS + 1
                k1   = 0
            else:                 # data from sfc to top
                ksfc = 1
                k1   = 1

            for k in range(LEVS):
                for i in range(IM):
                    totcnd = qcwat[i, k] + qcice[i, k] + qrain[i, k]
                    qsnow = 0.0
                    if totcnd > EPSQ:
                        # air density (rho), model mass thickness (cpath), temperature in c (tc)
                        rho = 0.1 * plyr[i, k] / \
                            (RD* tlyr[i, k] * (1.0 + EPS1*qlyr[i, k]))
                        cpath = abs(plvl[i, k+1] - plvl[i, k]) * (100000.0 / GRAV)
                        tc = tlyr[i, k] - T0C
    
                        # cloud water
                        # ---  effective radius (recwat) & total water path (cwatp):
                        #      assume monodisperse distribution of droplets (no factor of 1.5)
    
                        if qcwat[i, k] > 0.0:
                            recwat[i, k] = max(RECWmin,recw1*(rho*qcwat[i, k])**CEXP)
                            cwatp[i, k] = cpath * qcwat[i, k]           # cloud water path
    
                        # rain
                        # ---  effective radius (rerain) & total water path (rainp):
                        #      factor of 1.5 accounts for r**3/r**2 moments for exponentially
                        #      distributed drops in effective radius calculations
                        #      (from m.d. chou's code provided to y.-t. hou)
    
                        if qrain[i, k] > 0.0:
                            tem = CN0r0 * np.sqrt(np.sqrt(rho*qrain[i, k]))
                            rerain[i, k] = 1.5 * max(XMRmin, min(XMRmax, tem))
                            rainp [i, k] = cpath * qrain[i, k]           # rain water path
    
                        # snow (large ice) & cloud ice
                        # ---  effective radius (resnow) & total ice path (snowp) for snow, and
                        #      total ice path (cicep) for cloud ice:
                        #      factor of 1.5 accounts for r**3/r**2 moments for exponentially
                        #      distributed ice particles in effective radius calculations
                        #      separation of cloud ice & "snow" uses algorithm from subroutine gsmcolumn
    
                        pfac = 1.0
    
                        if qcice[i, k] > 0.0:
                            #  ---  mean particle size following houze et al. (jas, 1979, p. 160),
                            #       converted from fig. 5 plot of lamdas.  an analogous set of
                            #       relationships also shown by fig. 8 of ryan (bams, 1996, p. 66),
                            #       but with a variety of different relationships that parallel
                            #       the houze curves.
    
                            dum = max(0.05, min(1.0, np.exp(0.0564*tc)))
                            indexs = min(MDImax, max(MDImin, int(XMImax*dum)))
                            DUM = max(flgmin[i]*pfac, DUM)
    
                            #  ---  assumed number fraction of large ice to total (large & small) ice
                            #       particles, which is based on a general impression of the literature.
                            #       small ice are assumed to have a mean diameter of 50 microns.
    
                            if tc >= 0.0:
                                flarge = FLG1P0
                            else:
                                flarge = dum
    
                            xsimass = MASSI[MDImin] * (1.0 - flarge) / flarge
                            NLImax = 10.e3/np.sqrt(DUM)       #- Ver3
    
                            tem = rho * qcice[i, k]
                            nlice = tem / (xsimass +rrime(i,k)*MASSI[indexs])
    
                            #  ---  from subroutine gsmcolumn:
                            #       minimum number concentration for large ice of NLImin=10/m**3
                            #       at t>=0c.  done in order to prevent unrealistically small
                            #       melting rates and tiny amounts of snow from falling to
                            #       unrealistically warm temperatures.
    
                            if tc >= 0.0:
                                nlice = max(NLImin, nlice)
                            elif nlice > NLImax:
                                #  ---  ferrier 6/13/01:  prevent excess accumulation of ice
                                xli = (tem/NLImax - xsimass) / rrime[i, k]
    
                                if xli <= MASSI[450]:
                                    dsnow = 9.5885e5 * xli**0.42066
                                else:
                                    dsnow = 3.9751e6 * xli** 0.49870
    
                                indexs = min(MDImax, max(indexs, int(dsnow)))
                                nlice = tem / (xsimass + rrime[i, k]*MASSI[indexs])
    
                            if (plvl[i, ksfc] > 850.0 and plvl[i, k+k1] > 700.0 and indexs >= INDEXSmin):
                                qsnow = min(qcice[i, k], nlice*rrime[i, k]*MASSI[indexs]/rho)
    
                            qqcice = max(0.0, qcice[i, k]-qsnow)
                            cicep [i, k] = cpath * qqcice          # cloud ice path
                            resnow[i, k] = 1.5 * float(indexs)
                            snden [i, k] = SDENS(indexs) / rrime[i, k]   # 1/snow density
                            snowp [i, k] = cpath*qsnow             # snow path
    
    return cwatp, cicep, rainp, snowp, recwat, rerain, resnow, snden