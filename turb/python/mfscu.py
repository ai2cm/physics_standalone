import numpy as np
from routines import fpvs, gpvs

def mfscu(im,ix,km,kmscu,ntcw,ntrac1,delt,
          cnvflg,zl,zm,q1,t1,u1,v1,plyr,pix,
          thlx,thvx,thlvx,gdx,thetae,radj,
          krad,mrad,radmin,buo,xmfd,
          tcdo,qcdo,ucdo,vcdo,xlamde,
          g, gocp, elocp, el2orc,
          compare_dict):

    ce0 = 0.4
    cm = 1.0
    pgcon = 0.55
    qmin = 1e-8
    qlmin = 1e-12
    b1 = 0.45
    f1 = 0.15
    a1 = 0.12
    a2 = 0.50
    a11 = 0.2
    a22 = 1.0
    cldtime = 500.0
    actei = 0.7
    hvap = 2.5000e+6
    cp = 1.0046e+3
    eps = 2.8705e+2/4.6150e+2
    epsm1 = eps - 1.0
    fv = 4.6150e+2 / 2.8705e+2 - 1.0

    c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx = gpvs()
    
    totflg = True
    
    totflg = np.all(~cnvflg)

    if totflg:
        return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde
    
    dt2 = delt

    wd2 = np.zeros((im,km))
    qtx = np.zeros((im,km))
    hrad = np.zeros(im)
    krad1 = np.zeros(im,dtype=int)
    thld = np.zeros((im,km))
    qtd = np.zeros((im,km))
    thlvd = np.zeros(im)
    ra1 = np.zeros(im)
    ra2 = np.zeros(im)
    flg = np.zeros(im,dtype=bool)
    xlamdem = np.zeros((im,km-1))
    thvd = np.zeros(im)
    mrady = np.zeros(im,dtype=int)
    mradx = np.zeros(im,dtype=int)
    xlamavg = np.zeros(im)
    sumx = np.zeros(im)
    sigma = np.zeros(im)
    scaldfunc = np.zeros(im)

    for k in range(km):
        buo[:,k] = cnvflg*0.0 + ~cnvflg*buo[:,k]
        wd2[:,k] = cnvflg*0.0 + ~cnvflg*wd2[:,k]
        qtx[:,k] = cnvflg*(q1[:,k,0] + q1[:,k,ntcw-1]) + ~cnvflg*qtx[:,k]

    # np.testing.assert_array_equal(buo, compare_dict["buo"])
    # test = compare_dict["wd2"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(wd2, test)
    # test = compare_dict["qtx"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(qtx, test)

    hrad[:] = cnvflg*zm[range(0,im),krad[range(0,im)]]
    krad1[:] = cnvflg*(krad-1) + ~cnvflg*krad1

    # test = compare_dict["hrad"]
    # test[np.isnan(test)] = 0.0
    # np.testing.assert_array_equal(hrad,test)
    # np.testing.assert_array_equal(krad1,compare_dict["krad1"]-1)


    # for i in range(im):
    #     if cnvflg[i]:
    #         k = krad[i]
    #         tem = zm[i,k+1] - zm[i,k]
    #         tem1 = cldtime*radmin[i]/tem
    #         tem1 = max(tem1,-3.0)
    #         thld[i,k] = thlx[i,k] + tem1
    #         qtd[i,k] = qtx[i,k]
    #         thlvd[i] = thlvx[i,k] + tem1
    #         buo[i,k] = -g * tem1 / thvx[i,k]
    k = krad
    tem = zm[range(im),k+1] - zm[range(im),k]
    tem1 = cldtime*radmin/tem
    tem1 = np.maximum(tem1,-3.0)
    thld[range(im),k] = cnvflg*(thlx[range(im),k] + tem1) + ~cnvflg*(thld[range(im),k])
    qtd[range(im),k] = cnvflg*qtx[range(im),k] + ~cnvflg*qtd[range(im),k]
    thlvd[range(im)] = cnvflg*(thlvx[range(im),k] + tem1) + ~cnvflg*thlvd[range(im)]
    buo[range(im),k] = cnvflg*(-g * tem1/thvx[range(im),k]) + ~cnvflg*buo[range(im),k]

    # assert_test(thld, compare_dict["thld"])
    # assert_test(qtd, compare_dict["qtd"])
    # assert_test(thlvd, compare_dict["thlvd"])
    # assert_test(buo, compare_dict["buo"])

    ra1[:] = cnvflg*a1
    ra2[:] = cnvflg*a11

    # assert_test(ra1, compare_dict["ra1"])
    # assert_test(ra2, compare_dict["ra2"])


    k = krad
    tem = thetae[range(im),k] - thetae[range(im),k+1]
    tem1 = qtx[range(im),k] - qtx[range(im),k+1]
    evalu = np.logical_and(tem > 0.0,tem1 > 0.0)
    cteit = cp*tem/(hvap*tem1)
    evalu1 = evalu*(cteit > actei)

    ra1[:] = evalu1*a2 + ~evalu1*ra1
    ra2[:] = evalu1*a22 + ~evalu1*ra2

    # assert_test(ra1, compare_dict["ra1"])
    # assert_test(ra2, compare_dict["ra2"])

    radj[:] = cnvflg*(-ra2 * radmin) + ~cnvflg*radj
    
    # assert_test(radj,compare_dict["radj"])
    flg[:] = cnvflg
    mrad[:] = krad

    # assert_test(flg, compare_dict["flg"])
    # assert_test(mrad, compare_dict["mrad"]-1)

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(flg,k < krad)
        evalu1 = thlvd <= thlvx[:,k]
        mrad[:] = evalu*(evalu1*k + ~evalu1*mrad) + ~evalu*mrad
        flg[:] = evalu*(~evalu1*False + evalu1*flg) + ~evalu*flg

    # np.testing.assert_array_equal(flg, compare_dict["flg"])
    # np.testing.assert_array_equal(mrad, compare_dict["mrad"]-1)

    kk = krad - mrad
    evalu = kk < 1
    cnvflg[:] = cnvflg*(evalu*False + ~evalu*cnvflg) + ~cnvflg*cnvflg

    #np.testing.assert_array_equal(cnvflg, compare_dict["cnvflg"])

    totflg = True

    totflg = np.all(~cnvflg)

    if totflg:
        return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde

    for k in range(kmscu):
        # for i in range(im):
        #     if cnvflg[i]:
        #         dz = zl[i,k+1] - zl[i,k]
        #         if (k >= mrad[i]) and (k < krad[i]):
        #             if mrad[i] == 0:
        #                 ptem = 1.0 / (zm[i,k] + dz)
        #             else:
        #                 ptem = 1.0 / (zm[i,k] - zm[i,mrad[i]-1] + dz)
        #             tem = max(hrad[i] - zm[i,k] + dz, dz)
        #             ptem1 = 1.0/tem
        #             xlamde[i,k] = ce0 * (ptem+ptem1)
        #         else:
        #             xlamde[i,k] = ce0 / dz
        #         xlamdem[i,k] = cm * xlamde[i,k]
        dz = zl[:,k+1] - zl[:,k]
        ptem_a = 1.0 / (zm[:,k] + dz)
        ptem_b = 1.0 / (zm[:,k] - zm[range(im), mrad[range(im)]-1] + dz)
        tem = np.maximum(hrad - zm[:,k] + dz, dz)
        ptem1 = 1.0/tem

        evalu = np.logical_and(k >= mrad, k < krad)
        evalu1 = mrad == 0
        xlamde[:,k] = cnvflg*(evalu*(evalu1*(ce0 * (ptem_a+ptem1)) + ~evalu1*(ce0*(ptem_b+ptem1))) + ~evalu*(ce0 / dz)) + ~cnvflg*xlamde[:,k]
        xlamdem[:,k] = cnvflg*(cm * xlamde[:,k]) + ~cnvflg*(xlamdem[:,k])

    # assert_test(xlamde, compare_dict["xlamde"])
    # assert_test(xlamdem, compare_dict["xlamdem"])

    for k in range(kmscu-1,-1,-1):
        # for i in range(im):
        #     if cnvflg[i] and (k < krad[i]):
        #         dz = zl[i,k+1] - zl[i,k]
        #         tem = 0.5 * xlamde[i,k] * dz
        #         factor = 1.0 + tem

        #         thld[i,k] = ((1.0 - tem) * thld[i,k+1] + tem*(thlx[i,k] + thlx[i,k+1]))/factor
        #         qtd[i,k] = ((1.0-tem) * qtd[i,k+1] + tem*(qtx[i,k] + qtx[i,k+1]))/factor

        #         tld = thld[i,k] / pix[i,k]
        #         es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tld)
        #         qs = max(qmin, eps*es/(plyr[i,k] + epsm1*es))
        #         dq = qtd[i,k] - qs

        #         if dq > 0.0:
        #             gamma = el2orc * qs / (tld**2)
        #             qld = dq/ (1.0 + gamma)
        #             qtd[i,k] = qs + qld
        #             tem1 = 1.0 + fv * qs - qld
        #             thdm = thld[i,k] + pix[i,k] * elocp * qld
        #             thvd = thdm * tem1
        #         else:
        #             tem1 = 1.0 + fv * qtd[i,k]
        #             thvd = thld[i,k] * tem1
        #         buo[i,k] = g * (1.0 - thvd / thvx[i,k])

        evalu = np.logical_and(cnvflg, k < krad)
        dz = zl[:,k+1] - zl[:,k]
        tem = 0.5 * xlamde[:,k] * dz
        factor = 1.0 + tem

        thld[:,k] = evalu*(((1.0 - tem) * thld[:,k+1] + tem * \
                     (thlx[:,k] + thlx[:,k+1]))/factor) + \
                    ~evalu*thld[:,k]
        qtd[:,k] = evalu*(((1.0-tem)*qtd[:,k+1]+tem * \
                    (qtx[:,k] + qtx[:,k+1]))/factor) + \
                   ~evalu*qtd[:,k]
        tld = thld[:,k] / pix[:,k]
        es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tld)
        qs = evalu*(np.maximum(qmin, eps*es / (plyr[:,k] + epsm1*es)))
        dq = evalu*(qtd[:,k] - qs)

        gamma = el2orc * qs / (tld**2)
        where_are_nans = np.isnan(gamma)
        gamma[where_are_nans] = 0.0

        qld = dq / (1.0+gamma)
        qtd[:,k] = evalu*((dq > 0.0) * (qs+qld) + ~(dq > 0.0) * qtd[:,k]) + ~evalu*qtd[:,k]
        tem1 = 1.0 + fv * qs - qld
        thdm = thld[:,k] + pix[:,k] * elocp * qld
        thvd[:] = (dq>0.0) * (thdm * tem1)
        tem1 = 1.0 + fv * qtd[:,k]
        thvd[:] = ~(dq>0.0) * (thld[:,k] * tem1) + (dq > 0.0) * thvd[:]
        buo[:,k] = evalu*(g * (1.0 - thvd / thvx[:,k])) + ~evalu*buo[:,k]

    # assert_test(thld, compare_dict["thld"])
    # assert_test(qtd, compare_dict["qtd"])
    # assert_test(buo, compare_dict["buo"])

    bb1 = 2.0
    bb2 = 4.0

    k = krad1
    dz = zm[range(im),k+1] - zm[range(im),k]
    tem = 0.5 * bb1 * xlamde[range(im),k] * dz
    tem1 = bb2 * buo[range(im), k+1] * dz
    ptem1 = 1.0 + tem
    wd2[range(im), k] = cnvflg*(tem1/ptem1) + ~cnvflg*wd2[range(im),k]

    # assert_test(wd2, compare_dict["wd2"])

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(cnvflg,k < krad1)
        dz = zm[:,k+1] - zm[:,k]
        tem = 0.25 * bb1 * (xlamde[:,k] + xlamde[:,k+1])*dz
        tem1 = bb2 * buo[:,k+1] * dz
        ptem = (1.0 - tem) * wd2[:,k+1]
        ptem1 = 1.0 + tem
        wd2[:,k] = evalu*((ptem + tem1) / ptem1) + ~evalu*wd2[:,k]

    # assert_test(wd2, compare_dict["wd2"])


    flg[:] = cnvflg
    mrady[:] = mrad
    mradx[:] = flg*krad + ~flg*mradx

    # assert_test(flg, compare_dict["flg"])
    # assert_test(mrady, compare_dict["mrady"]-1)
    # assert_test(mradx, compare_dict["mradx"]-1)

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(flg, k < krad)
        evalu1 = wd2[:,k] > 0.0
        mradx[:] = evalu*(evalu1*k + ~evalu1*mradx) + ~evalu*mradx
        flg[:] = evalu*(~evalu1*False + evalu1*flg) + ~evalu*flg

    # assert_test(flg, compare_dict["flg"])
    # assert_test(mradx, compare_dict["mradx"]-1)

    evalu = np.logical_and(cnvflg,mrad < mradx)
    mrad[:] = evalu*mradx + ~evalu*mrad

    # assert_test(mrad,compare_dict["mrad"]-1)

    kk = krad - mrad

    evalu = np.logical_and(cnvflg,kk < 0)
    cnvflg[:] = evalu*False + ~evalu*cnvflg

    # assert_test(cnvflg,compare_dict["cnvflg"])

    totflg = np.all(~cnvflg)

    if totflg:
        return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde

    evalu = np.logical_and(cnvflg,mrady < mradx)
    for k in range(kmscu):
        dz = zl[:,k+1] - zl[:,k]
        evalu1 = np.logical_and(k >= mrad, k < krad)
        ptem = (mrad == 0) * (1.0/(zm[:,k] + dz)) + ~(mrad==0) * (1.0/(zm[:,k] - zm[range(im),mrad[range(im)]-1]+dz))
        tem = np.maximum(hrad-zm[:,k] + dz, dz)
        ptem1 = 1.0/tem
        xlamde[:,k] = evalu*(evalu1*(ce0 * (ptem+ptem1)) + ~evalu1*(ce0/dz)) + ~evalu*xlamde[:,k]
        xlamdem[:,k] = evalu*(cm * xlamde[:,k]) + ~evalu*xlamdem[:,k]

    # assert_test(xlamde,compare_dict["xlamde"])
    # assert_test(xlamdem,compare_dict["xlamdem"])

    xlamavg[:] = 0.0
    sumx[:] = 0.0

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(cnvflg,np.logical_and(k >= mrad, k < krad))
        dz = zl[:,k+1] - zl[:,k]
        xlamavg[:] = evalu*(xlamavg + xlamde[:,k] * dz) + ~evalu*xlamavg
        sumx[:] = evalu*(sumx + dz) + ~evalu*sumx

    # assert_test(xlamavg,compare_dict["xlamavg"])
    # assert_test(sumx,compare_dict["sumx"])


    xlamavg[:] = cnvflg*(xlamavg / sumx) + ~cnvflg*xlamavg
    xlamavg[np.isnan(xlamavg)] = 0.0

    #assert_test(xlamavg,compare_dict["xlamavg"])

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(cnvflg,np.logical_and(k >= mrad, k < krad))
        tem = (wd2[:,k] > 0) * np.sqrt(wd2[:,k])# + ~(wd2[:,k] > 0) * (0.0)
        xmfd[:,k] = evalu*(ra1 * tem) + ~evalu*xmfd[:,k]

    # xmfd[np.isnan(xmfd)] = 0.0
    # assert_test(xmfd,compare_dict["xmfd"])

    tem = 0.2 / xlamavg
    tem1 = 3.14 * tem * tem
    sigma[:] = cnvflg*(tem1 / (gdx * gdx)) + ~cnvflg*sigma
    sigma[:] = cnvflg*(np.maximum(sigma,0.001))  + ~cnvflg*sigma
    sigma[:] = cnvflg*(np.minimum(sigma,0.999)) + ~cnvflg*sigma

    # sigma[np.isnan(sigma)] = 0.0
    # assert_test(sigma,compare_dict["sigma"])

    evalu = sigma > ra1
    scaldfunc[:] = cnvflg*(evalu*(np.maximum(np.minimum((1.0-sigma)*(1.0-sigma),1.0),0.0)) + ~evalu*1.0) + ~cnvflg*scaldfunc

    # scaldfunc[np.isnan(scaldfunc)] = 0.0
    # assert_test(scaldfunc,compare_dict["scaldfunc"])

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(cnvflg,np.logical_and(k >= mrad, k < krad))
        xmfd[:,k] = evalu*(scaldfunc * xmfd[:,k]) + ~evalu*xmfd[:,k]
        dz = zl[:,k+1] - zl[:,k]
        xmmx = dz / dt2
        xmfd[:,k] = evalu*(np.minimum(xmfd[:,k],xmmx)) + ~evalu*xmfd[:,k]
    
    # xmfd[np.isnan(xmfd)] = 0.0
    # assert_test(xmfd,compare_dict["xmfd"])

    k = krad
    thld[range(im),k] = cnvflg*thlx[range(im),k] + ~cnvflg*thld[range(im),k]

    # assert_test(thld,compare_dict["thld"])

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(cnvflg,np.logical_and(k >= mrad, k < krad))
        dz = zl[:,k+1] - zl[:,k]
        tem = 0.5 * xlamde[:,k] * dz
        factor = 1.0 + tem

        thld[:,k] = evalu*(((1.0-tem)*thld[:,k+1] + tem*(thlx[:,k]+thlx[:,k+1]))/factor) + ~evalu*thld[:,k]
        qtd[:,k] = evalu*(((1.0-tem)*qtd[:,k+1] + tem*(qtx[:,k] + qtx[:,k+1]))/factor) + ~evalu*qtd[:,k]
        tld = thld[:,k] / pix[:,k]
        es = 0.01 * fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, tld)
        qs = np.maximum(qmin, eps * es / (plyr[:,k] + epsm1*es))
        dq = qtd[:,k] - qs

        gamma = el2orc * qs / (tld**2)
        qld = dq / (1.0 + gamma)
        qtd[:,k] = evalu*((dq > 0)* (qs + qld) + ~(dq > 0) * qtd[:,k]) + ~evalu*qtd[:,k]
        qcdo[:,k,0] = evalu*((dq>0)* qs + ~(dq>0) * qtd[:,k]) + ~evalu*qcdo[:,k,0]
        qcdo[:,k,ntcw-1] = evalu*((dq>0)* qld) + ~evalu*qcdo[:,k,ntcw-1]
        tcdo[:,k] = evalu*((dq > 0)*(tld + elocp * qld) + ~(dq > 0)*tld) + ~evalu*tcdo[:,k]

    # qtd[np.isnan(qtd)] = 0.0
    # qcdo[np.isnan(qcdo)] = 0.0
    # tcdo[np.isnan(tcdo)] = 0.0
    # thld[np.isnan(thld)] = 0.0
    # assert_test(qtd,compare_dict["qtd"])
    # assert_test(qcdo,compare_dict["qcdo"])
    # assert_test(tcdo,compare_dict["tcdo"])
    # assert_test(thld,compare_dict["thld"])

    for k in range(kmscu-1,-1,-1):
        evalu = np.logical_and(np.logical_and(cnvflg,k < krad), k >= mrad)
        dz = zl[:,k+1] - zl[:,k]
        tem = 0.5 * xlamdem[:,k] * dz
        factor = 1.0 + tem
        ptem = tem - pgcon
        ptem1 = tem + pgcon

        ucdo[:,k] = evalu*(((1.0-tem)*ucdo[:,k+1] + ptem*u1[:,k+1]+ptem1*u1[:,k])/factor) + ~evalu*ucdo[:,k]
        vcdo[:,k] = evalu*(((1.0-tem)*vcdo[:,k+1] + ptem*v1[:,k+1]+ptem1*v1[:,k])/factor) + ~evalu*vcdo[:,k]

    # ucdo[np.isnan(ucdo)] = 0.0
    # vcdo[np.isnan(vcdo)] = 0.0
    # assert_test(ucdo,compare_dict["ucdo"])
    # assert_test(vcdo,compare_dict["vcdo"])

    if (ntcw > 2):
        for n in range(1,ntcw-1):
            for k in range(kmscu-1,-1,-1):
                evalu = np.logical_and(np.logical_and(cnvflg,k<krad),k >= mrad)
                dz = zl[:,k+1] - zl[:,k]
                tem = 0.5 * xlamde[:,k] * dz
                factor = 1.0 + tem

                qcdo[:,k,n] = evalu*(((1.0-tem)*qcdo[:,k+1,n] + tem*(q1[:,k,n]+q1[:,k+1,n]))/factor) + ~evalu*qcdo[:,k,n]
    
    # qcdo[np.isnan(qcdo)] = 0.0
    # assert_test(qcdo,compare_dict["qcdo"])

    ndc = ntrac1 - ntcw

    if ndc > 0:
        for n in range(ntcw,ntrac1):
            for k in range(kmscu-1,-1,-1):
                evalu = np.logical_and(np.logical_and(cnvflg,k<krad),k>=mrad)
                dz = zl[:,k+1] - zl[:,k]
                tem = 0.5 * xlamde[:,k] * dz
                factor = 1.0 + tem
                qcdo[:,k,n] = evalu*(((1.0-tem)*qcdo[:,k+1,n]+tem*(q1[:,k,n]+q1[:,k+1,n]))/factor) + ~evalu*qcdo[:,k,n]

    # qcdo[np.isnan(qcdo)] = 0.0
    # assert_test(qcdo,compare_dict["qcdo"])
  

    return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde

def assert_test(input1, input2):
    temp1 = input2
    temp1[np.isnan(temp1)] = 0.0
    input1[np.isnan(input1)] = 0.0
    np.testing.assert_array_equal(input1,temp1)
    #np.testing.assert_array_almost_equal(input1,input2)
    return 0