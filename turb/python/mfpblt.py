import numpy as np
from routines import gpvs, fpvs


def mfpblt(im,ix,km,kmpbl,ntcw,ntrac1,delt, 
           cnvflg,zl,zm,q1,t1,u1,v1,plyr,pix,thlx,thvx,
           gdx,hpbl,kpbl,vpert,buo,xmf, 
           tcko,qcko,ucko,vcko,xlamue,
           g, gocp, elocp, el2orc,
           compare_dict):

    ce0 = 0.4
    cm = 1.0
    qmin = 1e-8
    qlmin = 1e-12
    alp = 1.0
    pgcon = 0.55
    a1 = 0.13
    b1 = 0.5
    f1 = 0.15
    fv = 4.6150e+2 / 2.8705e+2 - 1.0
    eps = 2.8705e+2 / 4.6150e+2
    epsm1 = eps - 1.0

    wu2 = np.zeros((im,km))
    qtx = np.zeros((im,km))
    xlamuem = np.zeros((im,km-1))
    thlu = np.zeros((im,km))
    qtu = np.zeros((im,km))

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i]

    if totflg:
        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue
    
    dt2 = delt

    for k in range(km):
        buo[:,k] = cnvflg*0.0 + ~cnvflg*buo[:,k]
        wu2[:,k] = cnvflg*0.0 + ~cnvflg*wu2[:,k]
        qtx[:,k] = cnvflg*(q1[:,k,0] + q1[:,k,ntcw-1]) + ~cnvflg*qtx[:,k]

    # Check values of buo, wu2, and qtx
    # np.testing.assert_array_equal(buo,compare_dict["buo"])
    # np.testing.assert_array_equal(wu2,compare_dict["wu2"])
    # np.testing.assert_array_equal(qtx,compare_dict["qtx"])

    ptem = alp * vpert
    ptem = np.minimum(ptem,3.0)

    thlu[:,0] = cnvflg* (thlx[:,0] + ptem)
    qtu[:,0] = cnvflg * qtx[:,0]
    buo[:,0] = cnvflg * (g*ptem/thvx[:,0])

    # np.testing.assert_array_equal(buo,compare_dict["buo"])
    # np.testing.assert_array_equal(thlu[:,0],compare_dict["thlu"][:,0])
    # np.testing.assert_array_equal(qtu[:,0],compare_dict["qtu"][:,0])

    for k in range(kmpbl):
        # for i in range(im):
        #     if cnvflg[i]:
        #         dz = zl[i,k+1] - zl[i,k]
        #         if k < kpbl[i]:
        #             ptem = 1/(zm[i,k] + dz)
        #             tem = max(hpbl[i]-zm[i,k]+dz, dz)
        #             ptem1 = 1/tem
        #             xlamue[i,k] = ce0 * (ptem+ptem1)
        #         else:
        #             xlamue[i,k] = ce0 / dz
        #         xlamuem[i,k] = cm * xlamue[i,k]
        dz = zl[:,k+1] - zl[:,k]
        ptem = 1/(zm[:,k] + dz)
        tem = np.maximum(hpbl[:] - zm[:,k] + dz, dz)
        ptem1 = 1/tem
        eval1 = k < kpbl
        xlamue[:,k] = cnvflg*(eval1*(ce0*(ptem+ptem1)) + ~eval1*(ce0 / dz)) + ~cnvflg*xlamue[:,k]
        xlamuem[:,k] = cnvflg*(cm*xlamue[:,k]) + ~cnvflg*(xlamuem[:,k])
    
    # Check values of xlamue and xlamuem
    # np.testing.assert_array_equal(xlamue,compare_dict["xlamue"])
    # np.testing.assert_array_equal(xlamuem,compare_dict["xlamuem"])

    c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx = gpvs()

    thvu = np.zeros(im)

    for k in range(1,kmpbl):
        # for i in range(im):
        #     if cnvflg[i]:
        #         dz = zl[i,k] - zl[i,k-1]
        #         tem = 0.5 * xlamue[i,k-1] * dz
        #         factor = 1.0 + tem

        #         thlu[i,k] = ((1.0 - tem) * thlu[i,k-1] + tem * \
        #                      (thlx[i,k-1] + thlx[i,k]))/factor
        #         qtu[i,k] = ((1.0-tem)*qtu[i,k-1]+tem * \
        #                     (qtx[i,k-1] + qtx[i,k]))/factor
                
        #         tlu = thlu[i,k] / pix[i,k]
        #         es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tlu)
        #         qs = max(qmin, eps*es / (plyr[i,k] + epsm1*es))
        #         dq = qtu[i,k] - qs
                
        #         if dq > 0.0:
        #             gamma = el2orc * qs / (tlu**2)
        #             qlu = dq / (1.0 + gamma)
        #             qtu[i,k] = qs + qlu
        #             tem1 = 1.0 + fv * qs - qlu
        #             thup = thlu[i,k] + pix[i,k] * elocp * qlu
        #             thvu = thup * tem1
        #         else:
        #             tem1 = 1.0 + fv * qtu[i,k]
        #             thvu = thlu[i,k] * tem1
        #         buo[i,k] = g * (thvu / thvx[i,k] - 1.0)
        dz = zl[:,k] - zl[:,k-1]
        tem = 0.5 * xlamue[:,k-1] * dz
        factor = 1.0 + tem

        thlu[:,k] = cnvflg*(((1.0 - tem) * thlu[:,k-1] + tem * \
                     (thlx[:,k-1] + thlx[:,k]))/factor) + \
                    ~cnvflg*thlu[:,k]
        qtu[:,k] = cnvflg*(((1.0-tem)*qtu[:,k-1]+tem * \
                    (qtx[:,k-1] + qtx[:,k]))/factor) + \
                   ~cnvflg*qtu[:,k]
        tlu = thlu[:,k] / pix[:,k]
        es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tlu)
        qs = cnvflg*(np.maximum(qmin, eps*es / (plyr[:,k] + epsm1*es)))
        dq = cnvflg*(qtu[:,k] - qs)

        gamma = el2orc * qs / (tlu**2)
        where_are_nans = np.isnan(gamma)
        gamma[where_are_nans] = 0.0

        qlu = dq / (1.0+gamma)
        qtu[:,k] = cnvflg*((dq > 0.0) * (qs+qlu) + ~(dq > 0.0) * qtu[:,k]) + ~cnvflg*qtu[:,k]
        tem1 = 1.0 + fv * qs - qlu
        thup = thlu[:,k] + pix[:,k] * elocp * qlu
        thvu[:] = (dq>0.0) * (thup * tem1)
        tem1 = 1.0 + fv * qtu[:,k]
        thvu[:] = ~(dq>0.0) * (thlu[:,k] * tem1) + (dq > 0.0) * thvu[:]
        buo[:,k] = cnvflg*(g * (thvu / thvx[:,k] - 1.0)) + ~cnvflg*buo[:,k]

    # np.testing.assert_array_equal(thlu,compare_dict["thlu"])
    # np.testing.assert_array_equal(qtu,compare_dict["qtu"])
    # np.testing.assert_array_equal(buo,compare_dict["buo"])

    bb1 = 2.0
    bb2 = 4.0

    dz = zm[:,0]
    tem = 0.5 * bb1*xlamue[:,0] * dz
    tem1 = bb2 * buo[:,0] * dz
    ptem1 = 1.0 + tem
    wu2[:,0] = cnvflg*(tem1/ptem1) + ~cnvflg*(wu2[:,0])

    # np.testing.assert_array_equal(wu2[:,0],compare_dict["wu2"][:,0])

    for k in range(1,kmpbl):
        # for i in range(im):
        #     if cnvflg[i]:
        #         dz = zm[i,k] - zm[i,k-1]
        #         tem = 0.25*bb1*(xlamue[i,k] + xlamue[i,k-1])*dz
        #         tem1 = bb2 * buo[i,k] * dz
        #         ptem = (1.0 - tem) * wu2[i,k-1]
        #         ptem1 = 1 + tem
        #         wu2[i,k] = (ptem + tem1) / ptem1
        dz = zm[:,k] - zm[:,k-1]
        tem = 0.25*bb1*(xlamue[:,k] + xlamue[:,k-1])*dz
        tem1 = bb2 * buo[:,k] * dz
        ptem = (1.0 - tem) * wu2[:,k-1]
        ptem1 = 1 + tem
        wu2[:,k] = cnvflg*((ptem + tem1) / ptem1) + ~cnvflg*wu2[:,k]\

    # np.testing.assert_array_equal(wu2,compare_dict["wu2"])
    
    flg = np.zeros(im,dtype=bool)
    kpbly = np.zeros(im,dtype=int)
    kpblx = np.zeros(im,dtype=int)
    rbup = np.zeros(im)
    rbdn = np.zeros(im)

    flg[:] = True
    kpbly[:] = kpbl
    
    flg[:] = ~cnvflg
    rbup[:] = cnvflg*(wu2[:,0]) + ~cnvflg*(rbup)

    # np.testing.assert_array_equal(flg,compare_dict["flg"])
    # np.testing.assert_array_equal(rbup,compare_dict["rbup"])

    for k in range(1,kmpbl):
        # for i in range(im):
        #     if ~flg[i]:
        #         rbdn[i] = rbup[i]
        #         rbup[i] = wu2[i,k]
        #         kpblx[i] = k
        #         flg[i] = rbup[i] < 0.0
        rbdn[:] = ~flg*(rbup) + flg*rbdn
        rbup[:] = ~flg*wu2[:,k] + flg*rbup
        kpblx[:] = ~flg*k + flg*kpblx
        flg[:] = ~flg*(rbup<0.0) + flg

    # np.testing.assert_array_equal(rbdn,compare_dict["rbdn"])
    # np.testing.assert_array_equal(rbup,compare_dict["rbup"])
    # np.testing.assert_array_equal(kpblx,compare_dict["kpblx"]-1)
    # np.testing.assert_array_equal(flg,compare_dict["flg"])

    hpblx = np.zeros(im)

    for i in range(im):
        if cnvflg[i]:
            k = kpblx[i]
            if rbdn[i] <= 0.0:
                rbint = 0.0
            elif rbup[i] >= 0.0:
                rbint = 1.0
            else:
                rbint = rbdn[i] / (rbdn[i] - rbup[i])
            hpblx[i] = zm[i,k-1] + rbint*(zm[i,k] - zm[i,k-1])
        # k = kpblx
        # rbint = (rbdn <= 0.0) * 0.0 + (rbup >= 0.0) * 1.0 + np.logical_and(~(rbdn<=0.0),~(rbup >= 0.0)) * (rbdn / (rbdn-rbup))
        # hpblx[:] = cnvflg*(zm[:,k-1] + rbint*(zm[:,k] - zm[:,k-1])) + ~cnvflg*hpblx

    #np.testing.assert_array_equal(hpblx,compare_dict["hpblx"])

    evalu = np.logical_and(cnvflg,kpbl > kpblx)

    kpbl[:] = evalu*kpblx + ~evalu*kpbl
    hpbl[:] = evalu*hpblx + ~evalu*hpbl

    # np.testing.assert_array_equal(kpbl,compare_dict["kpbl"]-1)
    # np.testing.assert_array_equal(hpbl,compare_dict["hpbl"])
    
    evalu = np.logical_and(cnvflg,kpbly > kpblx)
    for k in range(kmpbl):
        # for i in range(im):
        #     if cnvflg[i] and (kpbly[i] > kpblx[i]):
        #         dz = zl[i,k+1] - zl[i,k]
        #         if k < kpbl[i]:
        #             ptem = 1.0 / (zm[i,k] + dz)
        #             tem = max(hpbl[i]-zm[i,k]+dz, dz)
        #             ptem1 = 1.0/tem
        #             xlamue[i,k] = ce0 * (ptem+ptem1)
        #         else:
        #             xlamue[i,k] = ce0 / dz
        #         xlamuem[i,k] = cm * xlamue[i,k]
        
        dz = zl[:,k+1] - zl[:,k]
        ptem = 1.0 / (zm[:,k] + dz)
        tem = np.maximum(hpbl-zm[:,k]+ dz, dz)
        ptem1 = 1.0/tem
        evalu2 = k<kpbl
        xlamue[:,k] = evalu*(evalu2*(ce0 * (ptem+ptem1)) + ~evalu2*(ce0/dz)) + ~evalu*xlamue[:,k]
        xlamuem[:,k] = evalu*(cm*xlamue[:,k]) + ~evalu*xlamuem[:,k]


    # test = compare_dict["xlamue"]
    # where_are_nans = np.isnan(test)
    # test[where_are_nans] = 0.0
    # np.testing.assert_array_equal(xlamue,test)
    
    # test = compare_dict["xlamuem"]
    # where_are_nans = np.isnan(test)
    # test[where_are_nans] = 0.0
    # np.testing.assert_array_equal(xlamuem,test)

    xlamavg = np.zeros(im)
    sumx = np.zeros(im)

    for k in range(kmpbl):
        evalu = np.logical_and(cnvflg, k < kpbl)
        dz = zl[:,k+1] - zl[:,k]
        xlamavg[:] = evalu*(xlamavg + xlamue[:,k] * dz) + ~evalu*xlamavg
        sumx[:] = evalu*(sumx + dz) + ~evalu*sumx

    # np.testing.assert_array_equal(xlamavg,compare_dict["xlamavg"])
    # np.testing.assert_array_equal(sumx,compare_dict["sumx"])

    xlamavg[:] = cnvflg*(xlamavg/sumx) + ~cnvflg*xlamavg

    xlamavg[np.isnan(xlamavg)] = 0.0

    # np.testing.assert_array_equal(xlamavg,compare_dict["xlamavg"])

    for k in range(kmpbl):
        evalu = np.logical_and(cnvflg,k<kpbl)
        tem = np.sqrt(wu2[:,k])

        xmf[:,k] = evalu*(a1*tem) + ~evalu*0.0

    xmf[np.isnan(xmf)] = 0.0
    # np.testing.assert_array_equal(xmf,compare_dict["xmf"])

    sigma = np.zeros(im)

    tem = 0.2 / xlamavg
    tem1 = 3.14 * tem * tem
    sigma[:] = cnvflg*(tem1 / (gdx * gdx)) + ~cnvflg*sigma
    sigma[:] = np.maximum(sigma, 0.001) + ~cnvflg*sigma
    sigma[:] = np.minimum(sigma, 0.999) + ~cnvflg*sigma

    # np.testing.assert_array_equal(sigma,compare_dict["sigma"])

    scaldfunc = np.zeros(im)

    evalu = sigma > a1
    scaldfunc[:] = (1.0-sigma) * (1.0-sigma)
    scaldfunc[:] = cnvflg*(evalu*(np.maximum(np.minimum(scaldfunc,1.0),0.0)) + ~evalu*1.0) + ~cnvflg * 0.0

    # np.testing.assert_array_equal(scaldfunc,compare_dict["scaldfunc"])

    for k in range(kmpbl):
        evalu = np.logical_and(cnvflg,k<kpbl)
        xmf[:,k] = evalu*(scaldfunc*xmf[:,k]) + ~evalu*xmf[:,k]
        dz = zl[:,k+1] - zl[:,k]
        xmmx = dz/dt2
        xmf[:,k] = evalu*(np.minimum(xmf[:,k],xmmx)) + ~evalu*xmf[:,k]

    # xmf[np.isnan(xmf)] = 0.0
    # np.testing.assert_array_equal(xmf,compare_dict["xmf"])

    thlu[:,0] = cnvflg*(thlx[:,0]) + ~cnvflg*(thlu[:,0])

    # np.testing.assert_array_equal(thlu,compare_dict["thlu"])

    for k in range(1,kmpbl):
        # for i in range(im):
        #     if cnvflg[i] and (k <= kpbl[i]):
        #         dz = zl[i,k] - zl[i,k-1]
        #         tem = 0.5 * xlamue[i,k-1] * dz
        #         factor = 1.0 + tem

        #         thlu[i,k] = ((1.0-tem)*thlu[i,k-1] + tem*(thlx[i,k-1]+thlx[i,k]))/factor
        #         qtu[i,k] = ((1.0-tem)*qtu[i,k-1] + tem*(qtx[i,k-1] + qtx[i,k]))/factor

        #         tlu = thlu[i,k] / pix[i,k]
        #         es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tlu)
        #         qs = max(qmin, eps*es / (plyr[i,k] + epsm1*es))
        #         dq = qtu[i,k] - qs

        #         if(dq > 0.0):
        #             gamma = el2orc * qs / (tlu**2)
        #             qlu = dq / (1.0 + gamma)
        #             qtu[i,k] = qs + qlu
        #             qcko[i,k,0] = qs
        #             qcko[i,k,ntcw-1] = qlu
        #             tcko[i,k] = tlu + elocp * qlu
        #         else:
        #             qcko[i,k,0] = qtu[i,k]
        #             qcko[i,k,ntcw-1] = 0.0
        #             tcko[i,k] = tlu
        evalu = np.logical_and(cnvflg,k <= kpbl)
        dz = zl[:,k] - zl[:,k-1]
        tem = 0.5 * xlamue[:,k-1] * dz
        factor = 1.0 + tem

        thlu[:,k] = evalu*(((1.0-tem)*thlu[:,k-1] + tem*(thlx[:,k-1]+thlx[:,k]))/factor) + ~evalu*thlu[:,k]
        qtu[:,k] = evalu*(((1.0-tem)*qtu[:,k-1] + tem*(qtx[:,k-1] + qtx[:,k]))/factor) + ~evalu*qtu[:,k]

        tlu = thlu[:,k] / pix[:,k]
        es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tlu)
        qs = np.maximum(qmin, eps*es / (plyr[:,k] + epsm1*es))
        dq = qtu[:,k] - qs

        gamma = el2orc * qs / (tlu**2)
        qlu = dq / (1.0 + gamma)
        evalu2 = dq > 0.0
        qtu[:,k] = evalu*(evalu2*(qs + qlu) + ~evalu2*qtu[:,k]) + ~evalu*qtu[:,k]
        qcko[:,k,0] = evalu*(evalu2*qs + ~evalu2*qtu[:,k]) + ~evalu*qcko[:,k,0]
        qcko[:,k,ntcw-1] = evalu*(evalu2*qlu + ~evalu2*0.0) + ~evalu*qcko[:,k,ntcw-1]
        tcko[:,k] = evalu*(evalu2*(tlu + elocp * qlu) + ~evalu2*tlu) + ~evalu*tcko[:,k]

    # np.testing.assert_array_equal(thlu,compare_dict["thlu"])
    # np.testing.assert_array_equal(qtu,compare_dict["qtu"])
    # np.testing.assert_array_equal(qcko[:,:,:3],compare_dict["qcko"][:,:,:3])
    # test = compare_dict["tcko"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(tcko,test)

    for k in range(1,kmpbl):
        # for i in range(im):
        #     if cnvflg[i] and (k <= kpbl[i]):
        #         dz = zl[i,k] - zl[i,k-1]
        #         tem = 0.5 * xlamuem[i,k-1] * dz
        #         factor = 1.0 + tem
        #         ptem = tem + pgcon
        #         ptem1 = tem - pgcon
        #         ucko[i,k] = ((1.0-tem) * ucko[i,k-1] + ptem*u1[i,k] + ptem1*u1[i,k-1])/factor
        #         vcko[i,k] = ((1.0-tem) * vcko[i,k-1] + ptem*v1[i,k] + ptem1*v1[i,k-1])/factor
        evalu = np.logical_and(cnvflg,k<=kpbl)
        dz = zl[:,k] - zl[:,k-1]
        tem = 0.5 * xlamuem[:,k-1] * dz
        factor = 1.0 + tem
        ptem = tem + pgcon
        ptem1 = tem - pgcon
        ucko[:,k] = evalu*(((1.0-tem) * ucko[:,k-1] + ptem*u1[:,k] + ptem1*u1[:,k-1])/factor) + ~evalu*ucko[:,k]
        vcko[:,k] = evalu*(((1.0-tem) * vcko[:,k-1] + ptem*v1[:,k] + ptem1*v1[:,k-1])/factor) + ~evalu*vcko[:,k]

    # test = compare_dict["ucko"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(ucko,test)
    # test = compare_dict["vcko"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(vcko,test)

    if ntcw > 2:
        for n in range(1,ntcw-1):
            for k in range(2,kmpbl):
                evalu = np.logical_and(cnvflg,k <= kpbl)
                dz = zl[:,k] - zl[:,k-1]
                tem = 0.5 * xlamue[:,k-1] * dz
                factor = 1.0 + tem

                qcko[:,k,n] = evalu*(((1.0-tem) * qcko[:,k-1,n] + tem*(q1[:,k,n]+q1[:,k-1,n])/factor)) * ~evalu*qcko[:,k,n]

    # np.testing.assert_array_equal(qcko[:,:,1:ntcw-1],compare_dict["qcko"][:,:,1:ntcw-1])

    ndc = ntrac1 - ntcw

    if ndc > 0:
        for n in range(ntcw,ntrac1):
            for k in range(1,kmpbl):
                evalu = np.logical_and(cnvflg,k <= kpbl)
                dz = zl[:,k] - zl[:,k-1]
                tem = 0.5 * xlamue[:,k-1] * dz
                factor = 1.0 + tem

                qcko[:,k,n] = evalu*(((1.0-tem)*qcko[:,k-1,n] + tem*(q1[:,k,n]+q1[:,k-1,n]))/factor) + ~evalu*qcko[:,k,n]

    # np.testing.assert_array_equal(qcko[:,:,ntcw:ntrac1],compare_dict["qcko"][:,:,ntcw:ntrac1])
    return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue