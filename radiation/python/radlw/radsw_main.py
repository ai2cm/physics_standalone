from python.radlw.radsw_param import NGPTSW
import sys

sys.path.insert(0, "/Users/AndrewP/Documents/work/physics_standalone/radiation/python")
import numpy as np
from radphysparam import iswmode, iswrgas, icldflg, iswrate
from phys_const import con_g, con_cp, con_amd, con_amw, con_amo3, con_avgd
from radsw_param import ntbmx, ngptsw, nbdsw, maxgas, nblow, nbhgh


class RadSWClass:
    VTAGSW = "NCEP SW v5.1  Nov 2012 -RRTMG-SW v3.8"

    eps = 1.0e-6
    oneminus = 1.0 - eps
    bpade = 1.0 / 0.278
    stpfac = 296.0 / 1013.0
    ftiny = 1.0e-12
    flimit = 1.0e-20
    s0 = 1368.22

    # atomic weights for conversion from mass to volume mixing ratios
    amdw = con_amd / con_amw
    amdo3 = con_amd / con_amo3

    # band indices
    nspa = [9, 9, 9, 9, 1, 9, 9, 1, 9, 1, 0, 1, 9, 1]
    nspb = [1, 5, 1, 1, 1, 5, 1, 0, 1, 0, 0, 1, 5, 1]

    idxsfc = [1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1]  # band index for sfc flux
    idxebc = [5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 5]  # band index for cld prop

    nuvb = 27  # uv-b band index

    ipsdsw0 = 1

    def __init__(self, me, iovrsw, isubcsw, iswcliq, exp_tbl):

        self.iovrsw = iovrsw
        self.isubcsw = isubcsw
        self.iswcliq = iswcliq
        self.exp_tbl = exp_tbl

        expeps = 1.0e-20

        #
        # ===> ... begin here
        #
        if self.iovrsw < 0 or self.iovrsw > 3:
            print(
                "*** Error in specification of cloud overlap flag",
                f" IOVRSW={self.iovrsw} in RSWINIT !!",
            )

        if me == 0:
            print(f"- Using AER Shortwave Radiation, Version: {self.VTAGSW}")

            if iswmode == 1:
                print("   --- Delta-eddington 2-stream transfer scheme")
            elif iswmode == 2:
                print("   --- PIFM 2-stream transfer scheme")
            elif iswmode == 3:
                print("   --- Discrete ordinates 2-stream transfer scheme")

            if iswrgas <= 0:
                print("   --- Rare gases absorption is NOT included in SW")
            else:
                print("   --- Include rare gases N2O, CH4, O2, absorptions in SW")

            if self.isubcsw == 0:
                print(
                    "   --- Using standard grid average clouds, no ",
                    "   sub-column clouds approximation applied",
                )
            elif self.isubcsw == 1:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with a prescribed sequence of permutation seeds",
                )
            elif self.isubcsw == 2:
                print(
                    "   --- Using MCICA sub-colum clouds approximation ",
                    "   with provided input array of permutation seeds",
                )
            else:
                print(
                    "  *** Error in specification of sub-column cloud ",
                    f" control flag isubcsw = {self.isubcsw} !!",
                )

        #  --- ...  check cloud flags for consistency

        if (icldflg == 0 and self.iswcliq != 0) or (icldflg == 1 and self.iswcliq == 0):
            print(
                "*** Model cloud scheme inconsistent with SW",
                " radiation cloud radiative property setup !!",
            )

        if self.isubcsw == 0 and self.iovrsw > 2:
            if me == 0:
                print(
                    f"*** IOVRSW={self.iovrsw} is not available for",
                    " ISUBCSW=0 setting!!",
                )
                print("The program will use maximum/random overlap", " instead.")
            self.iovrsw = 1

        #  --- ...  setup constant factors for heating rate
        #           the 1.0e-2 is to convert pressure from mb to N/m**2

        if iswrate == 1:
            self.heatfac = con_g * 864.0 / con_cp  #   (in k/day)
        else:
            self.heatfac = con_g * 1.0e-2 / con_cp  #   (in k/second)

        #  --- ...  define exponential lookup tables for transmittance. tau is
        #           computed as a function of the tau transition function, and
        #           transmittance is calculated as a function of tau.  all tables
        #           are computed at intervals of 0.0001.  the inverse of the
        #           constant used in the Pade approximation to the tau transition
        #           function is set to bpade.

        self.exp_tbl[0] = 1.0
        self.exp_tbl[ntbmx] = expeps

        for i in range(ntbmx - 1):
            tfn = i / (ntbmx - i)
            tau = self.bpade * tfn
            self.exp_tbl[i] = np.exp(-tau)

    def return_initdata(self):
        outdict = {"heatfac": self.heatfac, "exp_tbl": self.exp_tbl}
        return outdict

    def swrad(
        self,
        plyr,
        plvl,
        tlyr,
        tlvl,
        qlyr,
        olyr,
        gasvmr,
        clouds,
        icseed,
        aerosols,
        sfcalb,
        dzlyr,
        delpin,
        de_lgth,
        cosz,
        solcon,
        NDAY,
        idxday,
        npts,
        nlay,
        nlp1,
        lprnt,
        lhswb,
        lhsw0,
        lflxprf,
        lfdncmp,
    ):
        s0fac = solcon / self.s0

        cldfmc = np.zeros((nlay, ngptsw))
        taug = np.zeros((nlay, ngptsw))
        taur = np.zeros((nlay, ngptsw))

        fxupc = np.zeros((nlp1, nbdsw))
        fxdnc = np.zeros((nlp1, nbdsw))
        fxup0 = np.zeros((nlp1, nbdsw))
        fxdn0 = np.zeros((nlp1, nbdsw))

        tauae = np.zeros((nlay, nbdsw))
        ssaae = np.zeros((nlay, nbdsw))
        asyae = np.zeros((nlay, nbdsw))
        taucw = np.zeros((nlay, nbdsw))
        ssacw = np.zeros((nlay, nbdsw))
        asycw = np.zeros((nlay, nbdsw))

        sfluxzen = np.zeros(ngptsw)

        cldfrc = np.zeros(nlay)
        delp = np.zeros(nlay)
        pavel = np.zeros(nlay)
        tavel = np.zeros(nlay)
        coldry = np.zeros(nlay)
        colmol = np.zeros(nlay)
        h2ovmr = np.zeros(nlay)
        o3vmr = np.zeros(nlay)
        temcol = np.zeros(nlay)
        cliqp = np.zeros(nlay)
        reliq = np.zeros(nlay)
        cicep = np.zeros(nlay)
        reice = np.zeros(nlay)
        cdat1 = np.zeros(nlay)
        cdat2 = np.zeros(nlay)
        cdat3 = np.zeros(nlay)
        cdat4 = np.zeros(nlay)
        cfrac = np.zeros(nlay)
        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)
        selffac = np.zeros(nlay)
        selffrac = np.zeros(nlay)
        rfdelp = np.zeros(nlay)
        dz = np.zeros(nlay)

        fnet = np.zeros(nlp1)
        flxdc = np.zeros(nlp1)
        flxuc = np.zeros(nlp1)
        flxd0 = np.zeros(nlp1)
        flxu0 = np.zeros(nlp1)

        albbm = np.zeros(2)
        albdf = np.zeros(2)
        sfbmc = np.zeros(2)
        sfbm0 = np.zeros(2)
        sfdfc = np.zeros(2)
        sfdf0 = np.zeros(2)

        colamt = np.zeros((nlay, maxgas))

        indfor = np.zeros(nlay)
        indself = np.zeros(nlay)
        jp = np.zeros(nlay)
        jt = np.zeros(nlay)
        jt1 = np.zeros(nlay)

        hswc = np.zeros((npts, nlay))
        cldtau = np.zeros((npts, nlay))

        upfxc_t = np.zeros(npts)
        upfx0_t = np.zeros(npts)
        dnfxc_t = np.zeros(npts)

        upfxc_s = np.zeros(npts)
        upfx0_s = np.zeros(npts)
        dnfxc_s = np.zeros(npts)
        dnfx0_s = np.zeros(npts)

        upfxc_f = np.zeros((npts, nlp1))
        upfx0_f = np.zeros((npts, nlp1))
        dnfxc_f = np.zeros((npts, nlp1))
        dnfx0_f = np.zeros((npts, nlp1))

        uvbf0 = np.zeros(npts)
        uvbfc = np.zeros(npts)
        nirbm = np.zeros(npts)
        nirdf = np.zeros(npts)
        visbm = np.zeros(npts)
        visdf = np.zeros(npts)

        hsw0 = np.zeros((npts, nlay))
        hswb = np.zeros((npts, nlay, nbdsw))

        ipseed = np.zeros(npts)

        if self.isubcsw == 1:
            for i in range(npts):
                ipseed[i] = self.ipsdsw0 + i
        elif self.isubcsw == 2:
            for i in range(npts):
                ipseed[i] = icseed[i]

        if lprnt:
            print(
                f"In radsw, isubcsw = {self.isubcsw}, ipsdsw0 = {self.ipsdsw0}, ipseed = {ipseed}"
            )

        for ipt in range(NDAY):
            j1 = idxday(ipt)

            cosz1 = cosz[j1]
            sntz1 = 1.0 / cosz[j1]
            ssolar = s0fac * cosz[j1]
            if self.iovrsw == 3:
                delgth = de_lgth[j1]  # clouds decorr-length

            # Prepare surface albedo: bm,df - dir,dif; 1,2 - nir,uvv.
            albbm[0] = sfcalb[j1, 0]
            albdf[0] = sfcalb[j1, 1]
            albbm[1] = sfcalb[j1, 2]
            albdf[1] = sfcalb[j1, 3]

            tem1 = 100.0 * con_g
            tem2 = 1.0e-20 * 1.0e3 * con_avgd

            for k in range(nlay):
                pavel[k] = plyr[j1, k]
                tavel[k] = tlyr[j1, k]
                delp[k] = delpin[j1, k]
                dz[k] = dzlyr[j1, k]

                h2ovmr[k] = max(
                    0.0, qlyr[j1, k] * self.amdw / (1.0 - qlyr[j1, k])
                )  # input specific humidity
                o3vmr[k] = max(0.0, olyr[j1, k] * self.amdo3)  # input mass mixing ratio

                tem0 = (1.0 - h2ovmr[k]) * con_amd + h2ovmr[k] * con_amw
                coldry[k] = tem2 * delp[k] / (tem1 * tem0 * (1.0 + h2ovmr[k]))
                temcol[k] = 1.0e-12 * coldry[k]

                colamt[k, 0] = max(0.0, coldry[k] * h2ovmr[k])  # h2o
                colamt[k, 1] = max(temcol[k], coldry[k] * gasvmr[j1, k, 0])  # co2
                colamt[k, 2] = max(0.0, coldry[k] * o3vmr[k])  # o3
                colmol[k] = coldry[k] + colamt[k, 0]

            if lprnt:
                if ipt == 1:
                    print(f" pavel={pavel}")
                    print(f" tavel={tavel}")
                    print(f" delp={delp}")
                    print(f" h2ovmr={h2ovmr*1000}")
                    print(f" o3vmr={o3vmr*1000000}")

            if iswrgas > 0:
                for k in range(nlay):
                    colamt[k, 3] = max(temcol[k], coldry[k] * gasvmr[j1, k, 1])  # n2o
                    colamt[k, 4] = max(temcol[k], coldry[k] * gasvmr[j1, k, 2])  # ch4
                    colamt[k, 5] = max(temcol[k], coldry[k] * gasvmr[j1, k, 3])  # o2
            else:
                for k in range(nlay):
                    colamt[k, 3] = temcol[k]  # n2o
                    colamt[k, 4] = temcol[k]  # ch4
                    colamt[k, 5] = temcol[k]  # o2

            for ib in range(nbdsw):
                for k in range(nlay):
                    tauae[k, ib] = aerosols[j1, k, ib, 0]
                    ssaae[k, ib] = aerosols[j1, k, ib, 1]
                    asyae[k, ib] = aerosols[j1, k, ib, 2]

            if self.iswcliq > 0:
                for k in range(nlay):
                    cfrac[k] = clouds[j1, k, 0]  # cloud fraction
                    cliqp[k] = clouds[j1, k, 1]  # cloud liq path
                    reliq[k] = clouds[j1, k, 2]  # liq partical effctive radius
                    cicep[k] = clouds[j1, k, 3]  # cloud ice path
                    reice[k] = clouds[j1, k, 4]  # ice partical effctive radius
                    cdat1[k] = clouds[j1, k, 5]  # cloud rain drop path
                    cdat2[k] = clouds[j1, k, 6]  # rain partical effctive radius
                    cdat3[k] = clouds[j1, k, 7]  # cloud snow path
                    cdat4[k] = clouds[j1, k, 8]  # snow partical effctive radius

            else:
                for k in range(nlay):
                    cfrac[k] = clouds[j1, k, 0]  # cloud fraction
                    cdat1[k] = clouds[j1, k, 1]  # cloud optical depth
                    cdat2[k] = clouds[j1, k, 2]  # cloud single scattering albedo
                    cdat3[k] = clouds[j1, k, 3]  # cloud asymmetry factor

        # Compute fractions of clear sky view:
        #    - random overlapping
        #    - max/ran overlapping
        #    - maximum overlapping

        zcf0 = 1.0
        zcf1 = 1.0
        if self.iovrsw == 0:  # random overlapping
            for k in range(nlay):
                zcf0 = zcf0 * (1.0 - cfrac[k])
        elif self.iovrsw == 1:  # max/ran overlapping
            for k in range(nlay):
                if cfrac[k] > self.ftiny:  # cloudy layer
                    zcf1 = min(zcf1, 1.0 - cfrac[k])
                elif zcf1 < 1.0:  # clear layer
                    zcf0 = zcf0 * zcf1
                    zcf1 = 1.0
            zcf0 = zcf0 * zcf1
        elif self.iovrsw >= 2:
            for k in range(nlay):
                zcf0 = min(zcf0, 1.0 - cfrac[k])  # used only as clear/cloudy indicator

        if zcf0 <= self.ftiny:
            zcf0 = 0.0
        if zcf0 > self.oneminus:
            zcf0 = 1.0
        zcf1 = 1.0 - zcf0

        # For cloudy sky column, call cldprop() to compute the cloud
        # optical properties for each cloudy layer.

        if zcf1 > 0.0:
            taucw, ssacw, asycw, cldfrc, cldfmc = self.cldprop(
                cfrac,
                cliqp,
                reliq,
                cicep,
                reice,
                cdat1,
                cdat2,
                cdat3,
                cdat4,
                zcf1,
                nlay,
                ipseed(j1),
                dz,
                delgth,
            )

            for k in range(nlay):
                cldtau[j1, k] = taucw[k, 9]
        else:
            cldfrc[:] = 0.0
            cldfmc[:, :] = 0.0
            for i in range(nbdsw):
                for k in range(nlay):
                    taucw[k, i] = 0.0
                    ssacw[k, i] = 0.0
                    asycw[k, i] = 0.0

        (
            laytrop,
            jp,
            jt,
            jt1,
            fac00,
            fac01,
            fac10,
            fac11,
            selffac,
            selffrac,
            indself,
            forfac,
            forfrac,
            indfor,
        ) = self.setcoef(pavel, tavel, h2ovmr, nlay, nlp1)

        sfluxzen, taug, taur = self.taumol(
            colamt,
            colmol,
            fac00,
            fac01,
            fac10,
            fac11,
            jp,
            jt,
            jt1,
            laytrop,
            forfac,
            forfrac,
            indfor,
            selffac,
            selffrac,
            indself,
            nlay,
        )

        if self.isubcsw <= 0:
            (
                fxupc,
                fxdnc,
                fxup0,
                fxdn0,
                ftoauc,
                ftoau0,
                ftoadc,
                fsfcuc,
                fsfcu0,
                fsfcdc,
                fsfcd0,
                sfbmc,
                sfdfc,
                sfbm0,
                sfdf0,
                suvbfc,
                suvbf0,
            ) = self.spcvrtc(
                ssolar,
                cosz1,
                sntz1,
                albbm,
                albdf,
                sfluxzen,
                cldfrc,
                zcf1,
                zcf0,
                taug,
                taur,
                tauae,
                ssaae,
                asyae,
                taucw,
                ssacw,
                asycw,
                nlay,
                nlp1,
            )
        else:
            (
                fxupc,
                fxdnc,
                fxup0,
                fxdn0,
                ftoauc,
                ftoau0,
                ftoadc,
                fsfcuc,
                fsfcu0,
                fsfcdc,
                fsfcd0,
                sfbmc,
                sfdfc,
                sfbm0,
                sfdf0,
                suvbfc,
                suvbf0,
            ) = self.spcvrtm(
                ssolar,
                cosz1,
                sntz1,
                albbm,
                albdf,
                sfluxzen,
                cldfmc,
                zcf1,
                zcf0,
                taug,
                taur,
                tauae,
                ssaae,
                asyae,
                taucw,
                ssacw,
                asycw,
                nlay,
                nlp1,
            )

        # Save outputs.
        #  --- ...  sum up total spectral fluxes for total-sky

        for k in range(nlp1):
            flxuc[k] = 0.0
            flxdc[k] = 0.0

            for ib in range(nbdsw):
                flxuc[k] = flxuc[k] + fxupc[k, ib]
                flxdc[k] = flxdc[k] + fxdnc[k, ib]

        # --- ...  optional clear sky fluxes

        if lhsw0 or lflxprf:
            for k in range(nlp1):
                flxu0[k] = 0.0
                flxd0[k] = 0.0

                for ib in range(nbdsw):
                    flxu0[k] = flxu0[k] + fxup0[k, ib]
                    flxd0[k] = flxd0[k] + fxdn0[k, ib]

        #  --- ...  prepare for final outputs

        for k in range(nlay):
            rfdelp[k] = self.heatfac / delp[k]

        if self.lfdncmp:
            # --- ...  optional uv-b surface downward flux
            uvbf0[j1] = suvbf0
            uvbfc[j1] = suvbfc

            # --- ...  optional beam and diffuse sfc fluxes
            nirbm[j1] = sfbmc[0]
            nirdf[j1] = sfdfc[0]
            visbm[j1] = sfbmc[1]
            visdf[j1] = sfdfc[1]

        #  --- ...  toa and sfc fluxes

        upfxc_t[j1] = ftoauc
        dnfxc_t[j1] = ftoadc
        upfx0_t[j1] = ftoau0

        upfxc_s[j1] = fsfcuc
        dnfxc_s[j1] = fsfcdc
        upfx0_s[j1] = fsfcu0
        dnfx0_s[j1] = fsfcd0

        #  --- ...  compute heating rates

        fnet[0] = flxdc[0] - flxuc[0]

        for k in range(1, nlp1):
            fnet[k] = flxdc[k] - flxuc[k]
            hswc[j1, k - 1] = (fnet[k] - fnet[k - 1]) * rfdelp[k - 1]

        # --- ...  optional flux profiles

        if lflxprf:
            for k in range(nlp1):
                upfxc_f[j1, k] = flxuc[k]
                dnfxc_f[j1, k] = flxdc[k]
                upfx0_f[j1, k] = flxu0[k]
                dnfx0_f[j1, k] = flxd0[k]

        # --- ...  optional clear sky heating rates

        if lhsw0:
            fnet[0] = flxd0[0] - flxu0[0]

            for k in range(1, nlp1):
                fnet[k] = flxd0[k] - flxu0[k]
                hsw0[j1, k - 1] = (fnet[k] - fnet[k - 1]) * rfdelp[k - 1]

        # --- ...  optional spectral band heating rates

        if lhswb:
            for mb in range(nbdsw):
                fnet[0] = fxdnc[0, mb] - fxupc[0, mb]

                for k in range(nlay):
                    fnet[k + 1] = fxdnc[k + 1, mb] - fxupc[k + 1, mb]
                    hswb[j1, k, mb] = (fnet[k + 1] - fnet[k]) * rfdelp[k]

    def cldprop(
        self,
        cfrac,
        cliqp,
        reliq,
        cicep,
        reice,
        cdat1,
        cdat2,
        cdat3,
        cdat4,
        cf1,
        nlay,
        ipseed,
        dz,
        delgth,
    ):
        cldfmc = np.zeros((nlay, ngptsw))
        taucw = np.zeros((nlay, nbdsw))
        ssacw = np.ones((nlay, nbdsw))
        asycw = np.zeros((nlay, nbdsw))
        cldfrc = np.zeros(nlay)

        ssaran = np.zeros(nbhgh - nblow)
        ssasnw = np.zeros(nbhgh - nblow)
        asyran = np.zeros(nbhgh - nblow)
        asysnw = np.zeros(nbhgh - nblow)
        tauliq = np.zeros(nbhgh - nblow)
        ssaliq = np.zeros(nbhgh - nblow)
        asyliq = np.zeros(nbhgh - nblow)

        if self.iswcliq > 0:
            for k in range(nlay):
                if cfrac[k] > self.ftiny:
                    cldran = cdat1[k]
                    cldsnw = cdat3[k]
                    refsnw = cdat4[k]
                    dgesnw = 1.0315 * refsnw  # for fu's snow formula
                    tauran = cldran * a0r

                    if cldsnw > 0.0 and refsnw > 10.0:
                        tausnw = cldsnw * 1.09087 * (a0s + a1s / dgesnw)  # fu's formula
                    else:
                        tausnw = 0.0

                    for ib in range(nblow - 1, nbhgh):
                        ssaran[ib] = tauran * (1.0 - b0r[ib])
                        ssasnw[ib] = tausnw * (1.0 - (b0s[ib] + b1s[ib] * dgesnw))
                        asyran[ib] = ssaran[ib] * c0r[ib]
                        asysnw[ib] = ssasnw[ib] * c0s[ib]

                    cldliq = cliqp[k]
                    cldice = cicep[k]
                    refliq = reliq[k]
                    refice = reice[k]

                    #  --- ...  calculation of absorption coefficients due to water clouds.

                    if cldliq <= 0.0:
                        for ib in range(nblow - 1, nbhgh):
                            tauliq[ib] = 0.0
                            ssaliq[ib] = 0.0
                            asyliq[ib] = 0.0
                    else:
                        factor = refliq - 1.5
                        index = max(1, min(57, int(factor))) - 1
                        fint = factor - float(index + 1)

                        if self.iswcliq == 1:
                            for ib in range(nblow - 1, nbhgh):
                                extcoliq = max(
                                    0.0,
                                    extliq1[index, ib]
                                    + fint
                                    * (extliq1[index + 1, ib] - extliq1[index, ib]),
                                )
                                ssacoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaliq1[index, ib]
                                        + fint
                                        * (ssaliq1[index + 1, ib] - ssaliq1[index, ib]),
                                    ),
                                )

                                asycoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyliq1[index, ib]
                                        + fint
                                        * (asyliq1[index + 1, ib] - asyliq1[index, ib]),
                                    ),
                                )

                                tauliq[ib] = cldliq * extcoliq
                                ssaliq[ib] = tauliq[ib] * ssacoliq
                                asyliq[ib] = ssaliq[ib] * asycoliq
                        elif self.iswcliq == 2:  # use updated coeffs
                            for ib in range(nblow - 1, nbhgh):
                                extcoliq = max(
                                    0.0,
                                    extliq2[index, ib]
                                    + fint
                                    * (extliq2[index + 1, ib] - extliq2[index, ib]),
                                )
                                ssacoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaliq2[index, ib]
                                        + fint
                                        * (ssaliq2[index + 1, ib] - ssaliq2[index, ib]),
                                    ),
                                )

                                asycoliq = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyliq2[index, ib]
                                        + fint
                                        * (asyliq2[index + 1, ib] - asyliq2[index, ib]),
                                    ),
                                )

                                tauliq[ib] = cldliq * extcoliq
                                ssaliq[ib] = tauliq[ib] * ssacoliq
                                asyliq[ib] = ssaliq[ib] * asycoliq

                    if cldice <= 0.0:
                        for ib in range(nblow - 1, nbhgh):
                            tauice[ib] = 0.0
                            ssaice[ib] = 0.0
                            asyice[ib] = 0.0
                    else:
                        #  --- ...  ebert and curry approach for all particle sizes though somewhat
                        #           unjustified for large ice particles
                        if self.iswcice == 1:
                            refice = min(130.0, max(13.0, refice))

                            for ib in range(nblow - 1, nbhgh):
                                ia = (
                                    self.idxebc[ib] - 1
                                )  # eb_&_c band index for ice cloud coeff

                                extcoice = max(0.0, abari[ia] + bbari[ia] / refice)
                                ssacoice = max(
                                    0.0, min(1.0, 1.0 - cbari[ia] - dbari[ia] * refice)
                                )
                                asycoice = max(
                                    0.0, min(1.0, ebari[ia] + fbari[ia] * refice)
                                )

                                tauice[ib] = cldice * extcoice
                                ssaice[ib] = tauice[ib] * ssacoice
                                asyice[ib] = ssaice[ib] * asycoice

                        #  --- ...  streamer approach for ice effective radius between 5.0 and 131.0 microns
                        elif self.iswcice == 2:
                            refice = min(131.0, max(5.0, refice))

                            factor = (refice - 2.0) / 3.0
                            index = max(1, min(42, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nblow - 1, nbhgh):
                                extcoice = max(
                                    0.0,
                                    extice2(index, ib)
                                    + fint
                                    * (extice2[index + 1, ib] - extice2[index, ib]),
                                )
                                ssacoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaice2[index, ib]
                                        + fint
                                        * (ssaice2[index + 1, ib] - ssaice2[index, ib]),
                                    ),
                                )
                                asycoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyice2[index, ib]
                                        + fint
                                        * (asyice2[index + 1, ib] - asyice2[index, ib]),
                                    ),
                                )

                                tauice[ib] = cldice * extcoice
                                ssaice[ib] = tauice[ib] * ssacoice
                                asyice[ib] = ssaice[ib] * asycoice

                        elif self.iswcice == 3:
                            dgeice = max(5.0, min(140.0, 1.0315 * refice))

                            factor = (dgeice - 2.0) / 3.0
                            index = max(1, min(45, int(factor))) - 1
                            fint = factor - float(index + 1)

                            for ib in range(nblow - 1, nbhgh):
                                extcoice = max(
                                    0.0,
                                    extice3[index, ib]
                                    + fint
                                    * (extice3[index + 1, ib] - extice3[index, ib]),
                                )
                                ssacoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ssaice3[index, ib]
                                        + fint
                                        * (ssaice3[index + 1, ib] - ssaice3[index, ib]),
                                    ),
                                )
                                asycoice = max(
                                    0.0,
                                    min(
                                        1.0,
                                        asyice3[index, ib]
                                        + fint
                                        * (asyice3[index + 1, ib] - asyice3[index, ib]),
                                    ),
                                )

                                tauice[ib] = cldice * extcoice
                                ssaice[ib] = tauice[ib] * ssacoice
                                asyice[ib] = ssaice[ib] * asycoice

                    for ib in range(nbdsw):
                        jb = nblow + ib - 2
                        taucw[k, ib] = tauliq[jb] + tauice[jb] + tauran + tausnw
                        ssacw[k, ib] = ssaliq[jb] + ssaice[jb] + ssaran[jb] + ssasnw[jb]
                        asycw[k, ib] = asyliq[jb] + asyice[jb] + asyran[jb] + asysnw[jb]
        else:

            for k in range(nlay):
                if cfrac[k] > self.ftiny:
                    for ib in range(nbdsw):
                        taucw[k, ib] = cdat1[k]
                        ssacw[k, ib] = cdat1[k] * cdat2[k]
                        asycw[k, ib] = ssacw[k, ib] * cdat3[k]

        # if physparam::isubcsw > 0, call mcica_subcol() to distribute
        #    cloud properties to each g-point.

        if self.isubcsw:
            cldf = cfrac
            cldf[cldf < self.ftiny] = 0.0

            lcloudy = self.mcica_subcol(cldf, nlay, ipseed, dz, delgth)

            for ig in range(ngptsw):
                for k in range(nlay):
                    if lcloudy[k, ig]:
                        cldfmc[k, ig] = 1.0
                    else:
                        cldfmc = 0.0
        else:
            for k in range(nlay):
                cldfrc[k] = cfrac[k] / cf1

        return taucw, ssacw, asycw, cldfrc, cldfmc
