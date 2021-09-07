import sys

sys.path.insert(0, "..")
import numpy as np
from radphysparam import iswmode, iswrgas, icldflg, iswrate
from phys_const import con_g, con_cp, con_amd, con_amw, con_amo3, con_avgd
from radsw.radsw_param import ntbmx, ngptsw, nbdsw, maxgas, nblow, nbhgh


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

    def setcoef(self, pavel, tavel, h2ovmr, nlay, nlp1):

        indself = np.zeros(nlay, dtype=np.int32)
        indfor = np.zeros(nlay, dtype=np.int32)
        jp = np.zeros(nlay, dtype=np.int32)
        jt = np.zeros(nlay, dtype=np.int32)
        jt1 = np.zeros(nlay, dtype=np.int32)

        fac00 = np.zeros(nlay)
        fac01 = np.zeros(nlay)
        fac10 = np.zeros(nlay)
        fac11 = np.zeros(nlay)
        selffac = np.zeros(nlay)
        selffrac = np.zeros(nlay)
        forfac = np.zeros(nlay)
        forfrac = np.zeros(nlay)

        laytrop = nlay

        for k in range(nlay):
            forfac[k] = pavel[k] * self.stpfac / (tavel[k] * (1.0 + h2ovmr[k]))

            #  --- ...  find the two reference pressures on either side of the
            #           layer pressure.  store them in jp and jp1.  store in fp the
            #           fraction of the difference (in ln(pressure)) between these
            #           two values that the layer pressure lies.

            plog = np.log(pavel[k])
            jp[k] = max(1, min(58, int(36.0 - 5.0 * (plog + 0.04)))) - 1
            jp1 = jp[k] + 1
            fp = 5.0 * (preflog(jp[k]) - plog)

            #  --- ...  determine, for each reference pressure (jp and jp1), which
            #          reference temperature (these are different for each reference
            #          pressure) is nearest the layer temperature but does not exceed it.
            #          store these indices in jt and jt1, resp. store in ft (resp. ft1)
            #          the fraction of the way between jt (jt1) and the next highest
            #          reference temperature that the layer temperature falls.

            tem1 = (tavel[k] - tref(jp[k])) / 15.0
            tem2 = (tavel[k] - tref(jp1)) / 15.0
            jt[k] = max(1, min(4, int(3.0 + tem1))) - 1
            jt1[k] = max(1, min(4, int(3.0 + tem2))) - 1
            ft = tem1 - float(jt[k] - 3)
            ft1 = tem2 - float(jt1[k] - 3)

            #  --- ...  we have now isolated the layer ln pressure and temperature,
            #           between two reference pressures and two reference temperatures
            #           (for each reference pressure).  we multiply the pressure
            #           fraction fp with the appropriate temperature fractions to get
            #           the factors that will be needed for the interpolation that yields
            #           the optical depths (performed in routines taugbn for band n).

            fp1 = 1.0 - fp
            fac10[k] = fp1 * ft
            fac00[k] = fp1 * (1.0 - ft)
            fac11[k] = fp * ft1
            fac01[k] = fp * (1.0 - ft1)

            #  --- ...  if the pressure is less than ~100mb, perform a different
            #           set of species interpolations.

            if plog > 4.56:

                laytrop = k

                #  --- ...  set up factors needed to separately include the water vapor
                #           foreign-continuum in the calculation of absorption coefficient.

                tem1 = (332.0 - tavel[k]) / 36.0
                indfor[k] = min(2, max(1, int(tem1)))
                forfrac[k] = tem1 - float(indfor[k])

                tem2 = (tavel[k] - 188.0) / 7.2
                indself[k] = min(9, max(1, int(tem2) - 7))
                selffrac[k] = tem2 - float(indself[k] + 7)
                selffac[k] = h2ovmr[k] * forfac[k]

            else:

                #  --- ...  set up factors needed to separately include the water vapor
                #           foreign-continuum in the calculation of absorption coefficient.

                tem1 = (tavel[k] - 188.0) / 36.0
                indfor[k] = 3
                forfrac[k] = tem1 - 1.0

                indself[k] = 0
                selffrac[k] = 0.0
                selffac[k] = 0.0

        return (
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
        )

    def spcvrtc(
        self,
        ssolar,
        cosz,
        sntz,
        albbm,
        albdf,
        sfluxzen,
        cldfrc,
        cf1,
        cf0,
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
    ):

        zcrit = 0.9999995  # thresold for conservative scattering
        zsr3 = np.sqrt(3.0)
        od_lo = 0.06
        eps1 = 1.0e-8

        fxupc = np.zeros((nlp1, nbdsw))
        fxdnc = np.zeros((nlp1, nbdsw))
        fxup0 = np.zeros((nlp1, nbdsw))
        fxdn0 = np.zeros((nlp1, nbdsw))

        sfbmc = np.zeros(2)
        sfdfc = np.zeros(2)
        sfbm0 = np.zeros(2)
        sfdf0 = np.zeros(2)

        ztaus = np.zeros(nlay)
        zssas = np.zeros(nlay)
        zasys = np.zeros(nlay)
        zldbt0 = np.zeros(nlay)

        zrefb = np.zeros(nlp1)
        zrefd = np.zeros(nlp1)
        ztrab = np.zeros(nlp1)
        ztrad = np.zeros(nlp1)
        ztdbt = np.zeros(nlp1)
        zldbt = np.zeros(nlp1)
        zfu = np.zeros(nlp1)
        zfd = np.zeros(nlp1)

        #  --- ...  loop over all g-points in each band

        for jg in range(ngptsw):
            jb = self.ngb[jg]
            ib = jb + 1 - nblow
            ibd = self.idxsfc[jb]

            zsolar = ssolar * sfluxzen[jg]

            #  --- ...  set up toa direct beam and surface values (beam and diff)

            ztdbt[nlp1 - 1] = 1.0
            ztdbt0 = 1.0

            zldbt[0] = 0.0
            if ibd != 0:
                zrefb[0] = albbm[ibd]
                zrefd[0] = albdf[ibd]
            else:
                zrefb[0] = 0.5 * (albbm[0] + albbm[1])
                zrefd[0] = 0.5 * (albdf[0] + albdf[1])

            ztrab[0] = 0.0
            ztrad[0] = 0.0

            # -# Compute clear-sky optical parameters, layer reflectance and
            #    transmittance.
            #    - Set up toa direct beam and surface values (beam and diff)
            #    - Delta scaling for clear-sky condition
            #    - General two-stream expressions for physparam::iswmode
            #    - Compute homogeneous reflectance and transmittance for both
            #      conservative and non-conservative scattering
            #    - Pre-delta-scaling clear and cloudy direct beam transmittance
            #    - Call swflux() to compute the upward and downward radiation
            #      fluxes

            for k in range(nlay - 1, -1, -1):
                kp = k + 1

                ztau0 = max(self.ftiny, taur[k, jg] + taug[k, jg] + tauae[k, ib])
                zssa0 = taur[k, jg] + tauae[k, ib] * ssaae[k, ib]
                zasy0 = asyae[k, ib] * ssaae[k, ib] * tauae[k, ib]
                zssaw = min(self.oneminus, zssa0 / ztau0)
                zasyw = zasy0 / max(self.ftiny, zssa0)

                #  --- ...  saving clear-sky quantities for later total-sky usage
                ztaus[k] = ztau0
                zssas[k] = zssa0
                zasys[k] = zasy0

                #  --- ...  delta scaling for clear-sky condition
                za1 = zasyw * zasyw
                za2 = zssaw * za1

                ztau1 = (1.0 - za2) * ztau0
                zssa1 = (zssaw - za2) / (1.0 - za2)
                zasy1 = zasyw / (1.0 + zasyw)  # to reduce truncation error
                zasy3 = 0.75 * zasy1

                #  --- ...  general two-stream expressions
                if iswmode == 1:
                    zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                    zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                    zgam3 = 0.5 - zasy3 * cosz
                elif iswmode == 2:  # pifm
                    zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                    zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                    zgam3 = 0.5 - zasy3 * cosz
                elif iswmode == 3:  # discrete ordinates
                    zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                    zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                    zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

                zgam4 = 1.0 - zgam3

                #  --- ...  compute homogeneous reflectance and transmittance

                if zssaw >= zcrit:  # for conservative scattering
                    za1 = zgam1 * cosz - zgam3
                    za2 = zgam1 * ztau1

                    #  --- ...  use exponential lookup table for transmittance, or expansion
                    #           of exponential for low optical depth

                    zb1 = min(ztau1 * sntz, 500.0)
                    if zb1 <= od_lo:
                        zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                    else:
                        ftind = zb1 / (self.bpade + zb1)
                        itind = ftind * ntbmx + 0.5
                        zb2 = self.exp_tbl[itind]

                    #      ...  collimated beam
                    zrefb[kp] = max(
                        0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                    )
                    ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                    #      ...  isotropic incidence
                    zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                    ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd[kp]))

                else:
                    za1 = zgam1 * zgam4 + zgam2 * zgam3
                    za2 = zgam1 * zgam3 + zgam2 * zgam4
                    zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                    zrk2 = 2.0 * zrk

                    zrp = zrk * cosz
                    zrp1 = 1.0 + zrp
                    zrm1 = 1.0 - zrp
                    zrpp1 = 1.0 - zrp * zrp
                    zrpp = np.sign(
                        max(self.flimit, abs(zrpp1)), zrpp1
                    )  # avoid numerical singularity
                    zrkg1 = zrk + zgam1
                    zrkg3 = zrk * zgam3
                    zrkg4 = zrk * zgam4

                    zr1 = zrm1 * (za2 + zrkg3)
                    zr2 = zrp1 * (za2 - zrkg3)
                    zr3 = zrk2 * (zgam3 - za2 * cosz)
                    zr4 = zrpp * zrkg1
                    zr5 = zrpp * (zrk - zgam1)

                    zt1 = zrp1 * (za1 + zrkg4)
                    zt2 = zrm1 * (za1 - zrkg4)
                    zt3 = zrk2 * (zgam4 + za1 * cosz)

                    #  --- ...  use exponential lookup table for transmittance, or expansion
                    #           of exponential for low optical depth

                    zb1 = min(zrk * ztau1, 500.0)
                    if zb1 <= od_lo:
                        zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                    else:
                        ftind = zb1 / (self.bpade + zb1)
                        itind = ftind * ntbmx + 0.5
                        zexm1 = self.exp_tbl[itind]

                    zexp1 = 1.0 / zexm1

                    zb2 = min(sntz * ztau1, 500.0)
                    if zb2 <= od_lo:
                        zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                    else:
                        ftind = zb2 / (self.bpade + zb2)
                        itind = ftind * ntbmx + 0.5
                        zexm2 = self.exp_tbl[itind]

                    zexp2 = 1.0 / zexm2
                    ze1r45 = zr4 * zexp1 + zr5 * zexm1

                    #      ...  collimated beam
                    if ze1r45 >= -eps1 and ze1r45 <= eps1:
                        zrefb[kp] = eps1
                        ztrab[kp] = zexm2
                    else:
                        zden1 = zssa1 / ze1r45
                        zrefb[kp] = max(
                            0.0,
                            min(1.0, (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2) * zden1),
                        )
                        ztrab[kp] = max(
                            0.0,
                            min(
                                1.0,
                                zexm2
                                * (
                                    1.0
                                    - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2) * zden1
                                ),
                            ),
                        )

                    #      ...  diffuse beam
                    zden1 = zr4 / (ze1r45 * zrkg1)
                    zrefd[kp] = max(0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1))
                    ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))

                #  --- ...  direct beam transmittance. use exponential lookup table
                #           for transmittance, or expansion of exponential for low
                #           optical depth

                zr1 = ztau1 * sntz
                if zr1 <= od_lo:
                    zexp3 = 1.0 - zr1 + 0.5 * zr1 * zr1
                else:
                    ftind = zr1 / (self.bpade + zr1)
                    itind = max(0, min(ntbmx, int(0.5 + ntbmx * ftind)))
                    zexp3 = self.exp_tbl[itind]

                ztdbt[k] = zexp3 * ztdbt[kp]
                zldbt[kp] = zexp3

                #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                #           (must use 'orig', unscaled cloud optical depth)

                zr1 = ztau0 * sntz
                if zr1 <= od_lo:
                    zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                else:
                    ftind = zr1 / (self.bpade + zr1)
                    itind = max(0, min(ntbmx, int(0.5 + ntbmx * ftind)))
                    zexp4 = self.exp_tbl[itind]

                zldbt0[k] = zexp4
                ztdbt0 = zexp4 * ztdbt0

            zfu, zfd = self.vrtqdr(zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1)

            #  --- ...  compute upward and downward fluxes at levels
            for k in range(nlp1):
                fxup0[k, ib] = fxup0[k, ib] + zsolar * zfu[k]
                fxdn0[k, ib] = fxdn0[k, ib] + zsolar * zfd[k]

            # --- ...  surface downward beam/diffused flux components
            zb1 = zsolar * ztdbt0
            zb2 = zsolar * (zfd[0] - ztdbt0)

            if ibd != 0:
                sfbm0[ibd] = sfbm0[ibd] + zb1
                sfdf0[ibd] = sfdf0[ibd] + zb2
            else:
                zf1 = 0.5 * zb1
                zf2 = 0.5 * zb2
                sfbm0[0] = sfbm0[0] + zf1
                sfdf0[0] = sfdf0[0] + zf2
                sfbm0[1] = sfbm0[1] + zf1
                sfdf0[1] = sfdf0[1] + zf2

            # -# Compute total sky optical parameters, layer reflectance and
            #    transmittance.
            #    - Set up toa direct beam and surface values (beam and diff)
            #    - Delta scaling for total-sky condition
            #    - General two-stream expressions for physparam::iswmode
            #    - Compute homogeneous reflectance and transmittance for
            #      conservative scattering and non-conservative scattering
            #    - Pre-delta-scaling clear and cloudy direct beam transmittance
            #    - Call swflux() to compute the upward and downward radiation fluxes

            if cf1 > self.eps:

                #  --- ...  set up toa direct beam and surface values (beam and diff)
                ztdbt0 = 1.0
                zldbt[0] = 0.0

                for k in range(nlay - 1, -1, -1):
                    kp = k + 1
                    zc0 = 1.0 - cldfrc[k]
                    zc1 = cldfrc[k]
                    if zc1 > self.ftiny:  # it is a cloudy-layer
                        ztau0 = ztaus[k] + taucw[k, ib]
                        zssa0 = zssas[k] + ssacw[k, ib]
                        zasy0 = zasys[k] + asycw[k, ib]
                        zssaw = min(self.oneminus, zssa0 / ztau0)
                        zasyw = zasy0 / max(self.ftiny, zssa0)

                        #  --- ...  delta scaling for total-sky condition
                        za1 = zasyw * zasyw
                        za2 = zssaw * za1

                        ztau1 = (1.0 - za2) * ztau0
                        zssa1 = (zssaw - za2) / (1.0 - za2)
                        zasy1 = zasyw / (1.0 + zasyw)
                        zasy3 = 0.75 * zasy1

                        #  --- ...  general two-stream expressions
                        if iswmode == 1:
                            zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                            zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                            zgam3 = 0.5 - zasy3 * cosz
                        elif iswmode == 2:  # pifm
                            zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                            zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                            zgam3 = 0.5 - zasy3 * cosz
                        elif iswmode == 3:  # discrete ordinates
                            zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                            zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                            zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

                        zgam4 = 1.0 - zgam3

                        zrefb1 = zrefb[kp]
                        zrefd1 = zrefd[kp]
                        ztrab1 = ztrab[kp]
                        ztrad1 = ztrad[kp]

                        #  --- ...  compute homogeneous reflectance and transmittance

                        if zssaw >= zcrit:  # for conservative scattering
                            za1 = zgam1 * cosz - zgam3
                            za2 = zgam1 * ztau1

                            #  --- ...  use exponential lookup table for transmittance, or expansion
                            #           of exponential for low optical depth

                            zb1 = min(ztau1 * sntz, 500.0)
                            if zb1 <= od_lo:
                                zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                            else:
                                ftind = zb1 / (self.bpade + zb1)
                                itind = ftind * ntbmx + 0.5
                                zb2 = self.exp_tbl[itind]

                            #      ...  collimated beam
                            zrefb[kp] = max(
                                0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                            )
                            ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                            #      ...  isotropic incidence
                            zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                            ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd[kp]))

                        else:  # for non-conservative scattering
                            za1 = zgam1 * zgam4 + zgam2 * zgam3
                            za2 = zgam1 * zgam3 + zgam2 * zgam4
                            zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                            zrk2 = 2.0 * zrk

                            zrp = zrk * cosz
                            zrp1 = 1.0 + zrp
                            zrm1 = 1.0 - zrp
                            zrpp1 = 1.0 - zrp * zrp
                            zrpp = np.sign(
                                max(self.flimit, abs(zrpp1)), zrpp1
                            )  # avoid numerical singularity
                            zrkg1 = zrk + zgam1
                            zrkg3 = zrk * zgam3
                            zrkg4 = zrk * zgam4

                            zr1 = zrm1 * (za2 + zrkg3)
                            zr2 = zrp1 * (za2 - zrkg3)
                            zr3 = zrk2 * (zgam3 - za2 * cosz)
                            zr4 = zrpp * zrkg1
                            zr5 = zrpp * (zrk - zgam1)

                            zt1 = zrp1 * (za1 + zrkg4)
                            zt2 = zrm1 * (za1 - zrkg4)
                            zt3 = zrk2 * (zgam4 + za1 * cosz)

                            #  --- ...  use exponential lookup table for transmittance, or expansion
                            #           of exponential for low optical depth

                            zb1 = min(zrk * ztau1, 500.0)
                            if zb1 <= od_lo:
                                zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                            else:
                                ftind = zb1 / (self.bpade + zb1)
                                itind = ftind * ntbmx + 0.5
                                zexm1 = self.exp_tbl[itind]

                            zexp1 = 1.0 / zexm1

                            zb2 = min(ztau1 * sntz, 500.0)
                            if zb2 <= od_lo:
                                zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                            else:
                                ftind = zb2 / (self.bpade + zb2)
                                itind = ftind * ntbmx + 0.5
                                zexm2 = self.exp_tbl[itind]

                            zexp2 = 1.0 / zexm2
                            ze1r45 = zr4 * zexp1 + zr5 * zexm1

                            #      ...  collimated beam
                            if ze1r45 >= -eps1 and ze1r45 <= eps1:
                                zrefb[kp] = eps1
                                ztrab[kp] = zexm2
                            else:
                                zden1 = zssa1 / ze1r45
                                zrefb[kp] = max(
                                    0.0,
                                    min(
                                        1.0,
                                        (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2)
                                        * zden1,
                                    ),
                                )
                                ztrab[kp] = max(
                                    0.0,
                                    min(
                                        1.0,
                                        zexm2
                                        * (
                                            1.0
                                            - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2)
                                            * zden1
                                        ),
                                    ),
                                )

                            #      ...  diffuse beam
                            zden1 = zr4 / (ze1r45 * zrkg1)
                            zrefd[kp] = max(
                                0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1)
                            )
                            ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))

                        #  --- ...  combine clear and cloudy contributions for total sky
                        #           and calculate direct beam transmittances

                        zrefb[kp] = zc0 * zrefb1 + zc1 * zrefb[kp]
                        zrefd[kp] = zc0 * zrefd1 + zc1 * zrefd[kp]
                        ztrab[kp] = zc0 * ztrab1 + zc1 * ztrab[kp]
                        ztrad[kp] = zc0 * ztrad1 + zc1 * ztrad[kp]

                        #  --- ...  direct beam transmittance. use exponential lookup table
                        #           for transmittance, or expansion of exponential for low
                        #           optical depth

                        zr1 = ztau1 * sntz
                        if zr1 <= od_lo:
                            zexp3 = 1.0 - zr1 + 0.5 * zr1 * zr1
                        else:
                            ftind = zr1 / (self.bpade + zr1)
                            itind = max(0, min(ntbmx, int(0.5 + ntbmx * ftind)))
                            zexp3 = self.exp_tbl[itind]

                        zldbt[kp] = zc0 * zldbt[kp] + zc1 * zexp3
                        ztdbt[k] = zldbt[kp] * ztdbt[kp]

                        #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                        #           (must use 'orig', unscaled cloud optical depth)

                        zr1 = ztau0 * sntz
                        if zr1 <= od_lo:
                            zexp4 = 1.0 - zr1 + 0.5 * zr1 * zr1
                        else:
                            ftind = zr1 / (self.bpade + zr1)
                            itind = max(0, min(ntbmx, int(0.5 + ntbmx * ftind)))
                            zexp4 = self.exp_tbl[itind]

                        ztdbt0 = (zc0 * zldbt0[k] + zc1 * zexp4) * ztdbt0

                    else:

                        #  --- ...  direct beam transmittance
                        ztdbt[k] = zldbt[kp] * ztdbt[kp]

                        #  --- ...  pre-delta-scaling clear and cloudy direct beam transmittance
                        ztdbt0 = zldbt0[k] * ztdbt0

                zfu, zfd = self.vrtqdr(
                    zrefb, zrefd, ztrab, ztrad, zldbt, ztdbt, nlay, nlp1
                )

                #  --- ...  compute upward and downward fluxes at levels
                for k in range(nlp1):
                    fxupc[k, ib] = fxupc[k, ib] + zsolar * zfu[k]
                    fxdnc[k, ib] = fxdnc[k, ib] + zsolar * zfd[k]

                # -# Process and save outputs.
                # --- ...  surface downward beam/diffused flux components
                zb1 = zsolar * ztdbt0
                zb2 = zsolar * (zfd[0] - ztdbt0)

                if ibd != 0:
                    sfbmc[ibd] = sfbmc[ibd] + zb1
                    sfdfc[ibd] = sfdfc[ibd] + zb2
                else:
                    zf1 = 0.5 * zb1
                    zf2 = 0.5 * zb2
                    sfbmc[0] = sfbmc[0] + zf1
                    sfdfc[0] = sfdfc[0] + zf2
                    sfbmc[1] = sfbmc[1] + zf1
                    sfdfc[1] = sfdfc[1] + zf2

        for ib in range(nbdsw):
            ftoadc = ftoadc + fxdn0[nlp1, ib]
            ftoau0 = ftoau0 + fxup0[nlp1, ib]
            fsfcu0 = fsfcu0 + fxup0[0, ib]
            fsfcd0 = fsfcd0 + fxdn0[0, ib]

        # --- ...  uv-b surface downward flux
        ibd = self.nuvb - nblow + 1
        suvbf0 = fxdn0[0, ibd]

        if cf1 <= self.eps:  # clear column, set total-sky=clear-sky fluxes
            for ib in range(nbdsw):
                for k in range(nlp1):
                    fxupc[k, ib] = fxup0[k, ib]
                    fxdnc[k, ib] = fxdn0[k, ib]

            ftoauc = ftoau0
            fsfcuc = fsfcu0
            fsfcdc = fsfcd0

            # --- ...  surface downward beam/diffused flux components
            sfbmc[0] = sfbm0[0]
            sfdfc[0] = sfdf0[0]
            sfbmc[1] = sfbm0[1]
            sfdfc[1] = sfdf0[1]

            # --- ...  uv-b surface downward flux
            suvbfc = suvbf0
        else:  # cloudy column, compute total-sky fluxes
            for ib in range(nbdsw):
                for k in range(nlp1):
                    fxupc[k, ib] = cf1 * fxupc[k, ib] + cf0 * fxup0[k, ib]
                    fxdnc[k, ib] = cf1 * fxdnc[k, ib] + cf0 * fxdn0[k, ib]

            for ib in range(nbdsw):
                ftoauc = ftoauc + fxupc[nlp1, ib]
                fsfcuc = fsfcuc + fxupc[0, ib]
                fsfcdc = fsfcdc + fxdnc[0, ib]

            # --- ...  uv-b surface downward flux
            suvbfc = fxdnc[0, ibd]

            # --- ...  surface downward beam/diffused flux components
            sfbmc[0] = cf1 * sfbmc[0] + cf0 * sfbm0[0]
            sfbmc[1] = cf1 * sfbmc[1] + cf0 * sfbm0[1]
            sfdfc[0] = cf1 * sfdfc[0] + cf0 * sfdf0[0]
            sfdfc[1] = cf1 * sfdfc[1] + cf0 * sfdf0[1]

        return (
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
        )

    def spcvrtm(
        self,
        ssolar,
        cosz,
        sntz,
        albbm,
        albdf,
        sfluxzen,
        cldfmc,
        cf1,
        cf0,
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
    ):
        #  ---  constant parameters:
        zcrit = 0.9999995  # thresold for conservative scattering
        zsr3 = np.sqrt(3.0)
        od_lo = 0.06
        eps1 = 1.0e-8

        fxupc = np.zeros((nlp1, nbdsw))
        fxdnc = np.zeros((nlp1, nbdsw))
        fxup0 = np.zeros((nlp1, nbdsw))
        fxdn0 = np.zeros((nlp1, nbdsw))

        sfbmc = np.zeros(2)
        sfdfc = np.zeros(2)
        sfbm0 = np.zeros(2)
        sfdf0 = np.zeros(2)

        ztaus = np.zeros(nlay)
        zssas = np.zeros(nlay)
        zasys = np.zeros(nlay)
        zldbt0 = np.zeros(nlay)

        zrefb = np.zeros(nlp1)
        zrefd = np.zeros(nlp1)
        ztrab = np.zeros(nlp1)
        ztrad = np.zeros(nlp1)
        ztdbt = np.zeros(nlp1)
        zldbt = np.zeros(nlp1)
        zfu = np.zeros(nlp1)
        zfd = np.zeros(nlp1)

        #  --- ...  loop over all g-points in each band
        for jg in range(ngptsw):

            jb = self.ngb[jg]
            ib = jb + 1 - nblow
            ibd = self.idxsfc[jb]  # spectral band index

            zsolar = ssolar * sfluxzen[jg]

            #  --- ...  set up toa direct beam and surface values (beam and diff)

            ztdbt[nlp1] = 1.0
            ztdbt0 = 1.0

            zldbt[0] = 0.0
            if ibd != 0:
                zrefb[0] = albbm[ibd]
                zrefd[0] = albdf[ibd]
            else:
                zrefb[0] = 0.5 * (albbm[0] + albbm[1])
                zrefd[0] = 0.5 * (albdf[0] + albdf[1])

            ztrab[0] = 0.0
            ztrad[0] = 0.0

            # -# Compute clear-sky optical parameters, layer reflectance and
            #    transmittance.
            #    - Set up toa direct beam and surface values (beam and diff)
            #    - Delta scaling for clear-sky condition
            #    - General two-stream expressions for physparam::iswmode
            #    - Compute homogeneous reflectance and transmittance for both
            #      conservative and non-conservative scattering
            #    - Pre-delta-scaling clear and cloudy direct beam transmittance
            #    - Call swflux() to compute the upward and downward radiation fluxes

            for k in range(nlay - 1, -1, -1):
                kp = k + 1

                ztau0 = max(self.ftiny, taur[k, jg] + taug[k, jg] + tauae[k, ib])
                zssa0 = taur[k, jg] + tauae[k, ib] * ssaae[k, ib]
                zasy0 = asyae[k, ib] * ssaae[k, ib] * tauae[k, ib]
                zssaw = min(self.oneminus, zssa0 / ztau0)
                zasyw = zasy0 / max(self.ftiny, zssa0)

                #  --- ...  saving clear-sky quantities for later total-sky usage
                ztaus[k] = ztau0
                zssas[k] = zssa0
                zasys[k] = zasy0

                #  --- ...  delta scaling for clear-sky condition
                za1 = zasyw * zasyw
                za2 = zssaw * za1

                ztau1 = (1.0 - za2) * ztau0
                zssa1 = (zssaw - za2) / (1.0 - za2)
                zasy1 = zasyw / (1.0 + zasyw)  # to reduce truncation error
                zasy3 = 0.75 * zasy1

                #  --- ...  general two-stream expressions
                if iswmode == 1:
                    zgam1 = 1.75 - zssa1 * (1.0 + zasy3)
                    zgam2 = -0.25 + zssa1 * (1.0 - zasy3)
                    zgam3 = 0.5 - zasy3 * cosz
                elif iswmode == 2:  # pifm
                    zgam1 = 2.0 - zssa1 * (1.25 + zasy3)
                    zgam2 = 0.75 * zssa1 * (1.0 - zasy1)
                    zgam3 = 0.5 - zasy3 * cosz
                elif iswmode == 3:  # discrete ordinates
                    zgam1 = zsr3 * (2.0 - zssa1 * (1.0 + zasy1)) * 0.5
                    zgam2 = zsr3 * zssa1 * (1.0 - zasy1) * 0.5
                    zgam3 = (1.0 - zsr3 * zasy1 * cosz) * 0.5

                zgam4 = 1.0 - zgam3

                #  --- ...  compute homogeneous reflectance and transmittance

                if zssaw >= zcrit:  # for conservative scattering
                    za1 = zgam1 * cosz - zgam3
                    za2 = zgam1 * ztau1

                    #  --- ...  use exponential lookup table for transmittance, or expansion
                    #           of exponential for low optical depth

                    zb1 = min(ztau1 * sntz, 500.0)
                    if zb1 <= od_lo:
                        zb2 = 1.0 - zb1 + 0.5 * zb1 * zb1
                    else:
                        ftind = zb1 / (self.bpade + zb1)
                        itind = ftind * ntbmx + 0.5
                        zb2 = self.exp_tbl[itind]

                    #      ...  collimated beam
                    zrefb[kp] = max(
                        0.0, min(1.0, (za2 - za1 * (1.0 - zb2)) / (1.0 + za2))
                    )
                    ztrab[kp] = max(0.0, min(1.0, 1.0 - zrefb[kp]))

                    #      ...  isotropic incidence
                    zrefd[kp] = max(0.0, min(1.0, za2 / (1.0 + za2)))
                    ztrad[kp] = max(0.0, min(1.0, 1.0 - zrefd[kp]))

                else:  # for non-conservative scattering
                    za1 = zgam1 * zgam4 + zgam2 * zgam3
                    za2 = zgam1 * zgam3 + zgam2 * zgam4
                    zrk = np.sqrt((zgam1 - zgam2) * (zgam1 + zgam2))
                    zrk2 = 2.0 * zrk

                    zrp = zrk * cosz
                    zrp1 = 1.0 + zrp
                    zrm1 = 1.0 - zrp
                    zrpp1 = 1.0 - zrp * zrp
                    zrpp = np.sign(
                        max(self.flimit, abs(zrpp1)), zrpp1
                    )  # avoid numerical singularity
                    zrkg1 = zrk + zgam1
                    zrkg3 = zrk * zgam3
                    zrkg4 = zrk * zgam4

                    zr1 = zrm1 * (za2 + zrkg3)
                    zr2 = zrp1 * (za2 - zrkg3)
                    zr3 = zrk2 * (zgam3 - za2 * cosz)
                    zr4 = zrpp * zrkg1
                    zr5 = zrpp * (zrk - zgam1)

                    zt1 = zrp1 * (za1 + zrkg4)
                    zt2 = zrm1 * (za1 - zrkg4)
                    zt3 = zrk2 * (zgam4 + za1 * cosz)

                    #  --- ...  use exponential lookup table for transmittance, or expansion
                    #           of exponential for low optical depth

                    zb1 = min(zrk * ztau1, 500.0)
                    if zb1 <= od_lo:
                        zexm1 = 1.0 - zb1 + 0.5 * zb1 * zb1
                    else:
                        ftind = zb1 / (self.bpade + zb1)
                        itind = ftind * ntbmx + 0.5
                        zexm1 = self.exp_tbl[itind]

                    zexp1 = 1.0 / zexm1

                    zb2 = min(sntz * ztau1, 500.0)
                    if zb2 <= od_lo:
                        zexm2 = 1.0 - zb2 + 0.5 * zb2 * zb2
                    else:
                        ftind = zb2 / (self.bpade + zb2)
                        itind = ftind * ntbmx + 0.5
                        zexm2 = self.exp_tbl[itind]

                    zexp2 = 1.0 / zexm2
                    ze1r45 = zr4 * zexp1 + zr5 * zexm1

                    #      ...  collimated beam
                    if ze1r45 >= -eps1 and ze1r45 <= eps1:
                        zrefb[kp] = eps1
                        ztrab[kp] = zexm2
                    else:
                        zden1 = zssa1 / ze1r45
                        zrefb[kp] = max(
                            0.0,
                            min(1.0, (zr1 * zexp1 - zr2 * zexm1 - zr3 * zexm2) * zden1),
                        )
                        ztrab[kp] = max(
                            0.0,
                            min(
                                1.0,
                                zexm2
                                * (
                                    1.0
                                    - (zt1 * zexp1 - zt2 * zexm1 - zt3 * zexp2) * zden1
                                ),
                            ),
                        )

                    #      ...  diffuse beam
                    zden1 = zr4 / (ze1r45 * zrkg1)
                    zrefd[kp] = max(0.0, min(1.0, zgam2 * (zexp1 - zexm1) * zden1))
                    ztrad[kp] = max(0.0, min(1.0, zrk2 * zden1))
