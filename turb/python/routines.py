import numpy as np
import math

def gpvs():
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax-xmin)/(nxpvs-1)
    c2xpvs = 1.0/xinc
    c1xpvs= 1.0 - (xmin*c2xpvs)
    tbpvs = np.zeros(nxpvs)
    for jx in range(nxpvs):
        x=xmin+(jx*xinc)
        tbpvs[jx] = fpvsx(x)

    return c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx


def fpvsx(t):
    con_ttp  = 2.7316e2
    con_cvap = 1.8460e3
    con_cliq = 4.1855e3
    con_hvap = 2.5000e6
    con_rv   = 4.6150e2
    con_csol = 2.1060e3
    con_hfus = 3.3358e5
    con_psat = 6.1078e2

    tliq   = con_ttp
    tice   = con_ttp - 20.0
    dldtl  = con_cvap - con_cliq
    heatl  = con_hvap
    xponal = -dldtl/con_rv
    xponbl = -dldtl/con_rv + heatl/(con_rv*con_ttp)
    dldti  = con_cvap - con_csol
    heati  = con_hvap + con_hfus
    xponai = -dldti/con_rv
    xponbi = -dldti/con_rv + heati/(con_rv*con_ttp)

    tr = con_ttp/t
    if t > tliq:
        fpvsx = con_psat * (tr**xponal) * math.exp(xponbl*(1.0-tr))
    elif t < tice:
        fpvsx = con_psat * (tr**xponai) * math.exp(xponbi*(1.0-tr))
    else:
        w = (t-tice)/(tliq-tice)
        pvl=con_psat*(tr**xponal)*math.exp(xponbl*(1.0-tr))
        pvi=con_psat*(tr**xponai)*math.exp(xponbi*(1.0-tr))
        fpvsx=w*pvl + (1.0-w) * pvi

    return fpvsx

def fpvs(c1xpvs, c2xpvs, nxpvs, tbpvs, fpvsx, t):
    # xj = min(max(c1xpvs+c2xpvs*t,1.0),nxpvs)
    # jx = min(xj, nxpvs-1.0)
    # jx = int(jx)
    # fpvs = tbpvs[jx-1] + (xj-jx) *(tbpvs[jx] - tbpvs[jx-1])
    
    xj = np.minimum(np.maximum(c1xpvs+c2xpvs*t,1.0),nxpvs)
    jx = np.minimum(xj,nxpvs-1.0)
    jx = jx.astype(int)

    fpvs = tbpvs[jx-1] + (xj-jx) *(tbpvs[jx] - tbpvs[jx-1])

    return fpvs