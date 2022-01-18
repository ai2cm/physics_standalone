import numpy as np
import math

GRAV = 9.80665e0 # Gravity
CA = 0.4          # Von karman Constant

CON_RV = 4.6150e2
CON_RD = 2.8705e2
RVRDM1 = CON_RV / CON_RD - 1.0

CHARNOCK = 0.014
Z0S_MAX = 0.317e-2
VIS = 1.4E-5
RNU = 1.51e-5
VISI = 1.0/VIS
LOG01 = math.log(0.01)
LOG05 = math.log(0.05)
LOG07 = math.log(0.07)

def stability(z1, snwdph, thv1, wind, z0max, ztmax, tvs,
              rb, fm, fh, fm10, fh2, cm, ch, stress, ustar):
    alpha = 5.0
    a0 = -3.975
    a1 = 12.32
    alpha4 = 4.0*alpha
    b1 = -7.755
    b2 = 6.041
    alpha2 = alpha + alpha
    beta = 1.0
    a0p = -7.941
    a1p = 24.75
    b1p = -8.705
    b2p = 7.899
    ztmin1 = -999.0

    z1i = 1.0 / z1

    tem1 = z0max / z1
    if abs(1.0 - tem1) > 1.0e-6:
        ztmax1 = - beta*math.log(tem1)/(alpha2 * (1.0 - tem1))
    else:
        ztmax1 = 99.0

    if z0max < 0.05 and snwdph < 10.0:
        ztmax1 = 99.0

    dtv = thv1 - tvs
    adtv = max(abs(dtv), 0.001)
    if dtv >= 0.0:
        dtv = abs(adtv)
    else:
        dtv = -abs(adtv)
    rb = max(-5000.0, (GRAV+GRAV) * dtv * z1 / ((thv1 + tvs) * wind * wind))
    tem1 = 1.0 / z0max
    tem2 = 1.0 / ztmax
    fm = math.log((z0max + z1) * tem1)
    fh = math.log((ztmax + z1) * tem2)
    fm10 = math.log((z0max + 10.0) * tem1)
    fh2 = math.log((ztmax + 2.0) * tem2)
    hlinf = rb * fm * fm / fh
    hlinf = min(max(hlinf, ztmin1), ztmax1)

    if dtv >= 0.0:
        hl1 = hlinf
        if hlinf > 0.25:
            tem1 = hlinf * z1i
            hl0inf = z0max * tem1
            hltinf = ztmax * tem1
            aa = math.sqrt(1.0 + alpha4 * hlinf)
            aa0 = math.sqrt(1.0 + alpha4 * hl0inf)
            bb = aa
            bb0 = math.sqrt(1.0 + alpha4 * hltinf)
            pm = aa0 - aa + math.log((aa+1.0)/(aa0+1.0))
            ph = bb0 - bb + math.log((bb+1.0)/(bb0+1.0))
            fms = fm - pm
            fhs = fh - ph
            hl1 = fms * fms * rb / fhs
            hl1 = min(max(hl1,ztmin1), ztmax1)

        tem1 = hl1 * z1i
        hl0   = z0max * tem1
        hlt   = ztmax * tem1
        aa    = math.sqrt(1. + alpha4 * hl1)
        aa0   = math.sqrt(1. + alpha4 * hl0)
        bb    = aa
        bb0   = math.sqrt(1. + alpha4 * hlt)
        pm    = aa0 - aa + math.log( (1.0+aa)/(1.0+aa0) )
        ph    = bb0 - bb + math.log( (1.0+bb)/(1.0+bb0) )
        hl110 = hl1 * 10. * z1i
        hl110 = min(max(hl110, ztmin1), ztmax1)
        aa    = math.sqrt(1. + alpha4 * hl110)
        pm10  = aa0 - aa + math.log( (1.0+aa)/(1.0+aa0) )
        hl12  = (hl1+hl1) * z1i
        hl12  = min(max(hl12,ztmin1),ztmax1)
        bb    = math.sqrt(1. + alpha4 * hl12)
        ph2   = bb0 - bb + math.log( (1.0+bb)/(1.0+bb0) )

    else:
        olinf = z1 / hlinf
        tem1  = 50.0 * z0max
        if abs(olinf) <= tem1:
            hlinf = -z1 / tem1
            hlinf = min(max(hlinf,ztmin1),ztmax1)
        if hlinf >= -0.5:
            hl1   = hlinf
            pm    = (a0  + a1*hl1)  * hl1   / (1.+ (b1+b2*hl1)  *hl1)
            ph    = (a0p + a1p*hl1) * hl1   / (1.+ (b1p+b2p*hl1)*hl1)
            hl110 = hl1 * 10. * z1i
            hl110 = min(max(hl110, ztmin1), ztmax1)
            pm10  = (a0 + a1*hl110) * hl110 / (1.+(b1+b2*hl110)*hl110)
            hl12  = (hl1+hl1) * z1i
            hl12  = min(max(hl12, ztmin1), ztmax1)
            ph2   = (a0p + a1p*hl12) * hl12 / (1.+(b1p+b2p*hl12)*hl12)
        else:                       # hlinf < 0.05
            hl1   = -hlinf
            tem1  = 1.0 / math.sqrt(hl1)
            pm    = math.log(hl1) + 2. * math.sqrt(tem1) - .8776
            ph    = math.log(hl1) + .5 * tem1 + 1.386
            hl110 = hl1 * 10. * z1i
            hl110 = min(max(hl110, ztmin1), ztmax1)
            pm10  = math.log(hl110) + 2.0 / math.sqrt(math.sqrt(hl110)) - .8776
            hl12  = (hl1+hl1) * z1i
            hl12  = min(max(hl12, ztmin1), ztmax1)
            ph2   = math.log(hl12) + 0.5 / math.sqrt(hl12) + 1.386

    fm        = fm - pm
    fh        = fh - ph
    fm10      = fm10 - pm10
    fh2       = fh2 - ph2
    cm        = CA * CA / (fm * fm)
    ch        = CA * CA / (fm * fh)
    tem1      = 0.00001/z1
    cm        = max(cm, tem1)
    ch        = max(ch, tem1)
    stress    = cm * wind * wind
    ustar     = math.sqrt(stress)

    return rb, fm, fh, fm10, fh2, cm, ch, stress, ustar


def znot_m_v6(uref):
    p13 = -1.296521881682694e-02
    p12 =  2.855780863283819e-01
    p11 = -1.597898515251717e+00
    p10 = -8.396975715683501e+00
    p25 =  3.790846746036765e-10
    p24 =  3.281964357650687e-09
    p23 =  1.962282433562894e-07
    p22 = -1.240239171056262e-06
    p21 =  1.739759082358234e-07
    p20 =  2.147264020369413e-05
    p35 =  1.840430200185075e-07
    p34 = -2.793849676757154e-05
    p33 =  1.735308193700643e-03
    p32 = -6.139315534216305e-02
    p31 =  1.255457892775006e+00
    p30 = -1.663993561652530e+01
    p40 =  4.579369142033410e-04

    if uref >= 0.0 and  uref <= 6.5:
        znotm = math.exp(p10 + uref * (p11 + uref * (p12 + uref*p13))) 
    elif uref > 6.5 and uref <= 15.7:
        znotm = p20 + uref * (p21 + uref * (p22 + uref * (p23
                    + uref * (p24 + uref * p25))))
    elif uref > 15.7 and uref <= 53.0:
        znotm = math.exp( p30 + uref * (p31 + uref * (p32 + uref * (p33
                        + uref * (p34 + uref * p35)))))
    elif uref > 53.0:
        znotm = p40
    else:
        print('Wrong input uref value:',uref)

    return znotm

def znot_t_v6(uref):
    p00 = 1.100000000000000e-04
    p15 = -9.144581627678278e-10
    p14 =  7.020346616456421e-08
    p13 = -2.155602086883837e-06
    p12 =  3.333848806567684e-05
    p11 = -2.628501274963990e-04
    p10 =  8.634221567969181e-04
    p25 = -8.654513012535990e-12
    p24 =  1.232380050058077e-09
    p23 = -6.837922749505057e-08
    p22 =  1.871407733439947e-06
    p21 = -2.552246987137160e-05
    p20 =  1.428968311457630e-04
    p35 =  3.207515102100162e-12
    p34 = -2.945761895342535e-10
    p33 =  8.788972147364181e-09
    p32 = -3.814457439412957e-08
    p31 = -2.448983648874671e-06
    p30 =  3.436721779020359e-05
    p45 = -3.530687797132211e-11
    p44 =  3.939867958963747e-09
    p43 = -1.227668406985956e-08
    p42 = -1.367469811838390e-05
    p41 =  5.988240863928883e-04
    p40 = -7.746288511324971e-03
    p56 = -1.187982453329086e-13
    p55 =  4.801984186231693e-11
    p54 = -8.049200462388188e-09
    p53 =  7.169872601310186e-07
    p52 = -3.581694433758150e-05
    p51 =  9.503919224192534e-04
    p50 = -1.036679430885215e-02
    p60 =  4.751256171799112e-05

    if uref >= 0.0 and uref < 5.9:
        znott = p00
    elif uref >= 5.9 and uref <= 15.4:
        znott = p10 + uref * (p11 + uref * (p12 + uref * (p13
                    + uref * (p14 + uref * p15))))
    elif uref > 15.4 and uref <= 21.6:
        znott = p20 + uref * (p21 + uref * (p22 + uref * (p23 
                    + uref * (p24 + uref * p25))))
    elif uref > 21.6 and uref <= 42.2:
        znott = p30 + uref * (p31 + uref * (p32 + uref * (p33 
                    + uref * (p34 + uref * p35))))
    elif uref > 42.2 and uref <= 53.3:
        znott = p40 + uref * (p41 + uref * (p42 + uref * (p43 
                    + uref * (p44 + uref * p45))))
    elif uref > 53.3 and uref <= 80.0:
        znott = p50 + uref * (p51 + uref * (p52 + uref * (p53 
                    + uref * (p54 + uref * (p55 + uref * p56)))))
    elif uref > 80.0:
        znott = p60
    else:
        print("Wrong input uref value", uref)
        
    return znott

def znot_m_v7(uref):
    p13 = -1.296521881682694e-02,
    p12 =  2.855780863283819e-01
    p11 = -1.597898515251717e+00
    p10 = -8.396975715683501e+00
    p25 =  3.790846746036765e-10
    p24 =  3.281964357650687e-09
    p23 =  1.962282433562894e-07
    p22 = -1.240239171056262e-06
    p21 =  1.739759082358234e-07
    p20 =  2.147264020369413e-05
    p35 =  1.897534489606422e-07
    p34 = -3.019495980684978e-05
    p33 =  1.931392924987349e-03
    p32 = -6.797293095862357e-02
    p31 =  1.346757797103756e+00
    p30 = -1.707846930193362e+01
    p40 =  3.371427455376717e-04

    if uref >= 0.0 and  uref <= 6.5:
        znotm = math.exp( p10 + uref * (p11 + uref * (p12 + uref * p13)))
    elif uref > 6.5 and uref <= 15.7:
        znotm = p20 + uref * (p21 + uref * (p22 + uref * (p23
                   + uref * (p24 + uref * p25))))
    elif uref > 15.7 and uref <= 53.0:
        znotm = math.exp( p30 + uref * (p31 + uref * (p32 + uref * (p33
                        + uref * (p34 + uref * p35)))))
    elif uref > 53.0:
        znotm = p40
    else:
        print('Wrong input uref value:',uref)

    return znotm

def znot_t_v7(uref):
    p00 =  1.100000000000000e-04
    p15 = -9.193764479895316e-10
    p14 =  7.052217518653943e-08
    p13 = -2.163419217747114e-06
    p12 =  3.342963077911962e-05
    p11 = -2.633566691328004e-04
    p10 =  8.644979973037803e-04
    p25 = -9.402722450219142e-12
    p24 =  1.325396583616614e-09
    p23 = -7.299148051141852e-08
    p22 =  1.982901461144764e-06
    p21 = -2.680293455916390e-05
    p20 =  1.484341646128200e-04
    p35 =  7.921446674311864e-12
    p34 = -1.019028029546602e-09
    p33 =  5.251986927351103e-08
    p32 = -1.337841892062716e-06
    p31 =  1.659454106237737e-05
    p30 = -7.558911792344770e-05
    p45 = -2.694370426850801e-10
    p44 =  5.817362913967911e-08
    p43 = -5.000813324746342e-06
    p42 =  2.143803523428029e-04
    p41 = -4.588070983722060e-03
    p40 =  3.924356617245624e-02
    p56 = -1.663918773476178e-13
    p55 =  6.724854483077447e-11
    p54 = -1.127030176632823e-08
    p53 =  1.003683177025925e-06
    p52 = -5.012618091180904e-05
    p51 =  1.329762020689302e-03
    p50 = -1.450062148367566e-02
    p60 =  6.840803042788488e-05

    if uref >= 0.0 and uref < 5.9 :
           znott = p00
    elif uref >= 5.9 and uref <= 15.4:
        znott = p10 + uref * (p11 + uref * (p12 + uref * (p13
                    + uref * (p14 + uref * p15))))
    elif uref > 15.4 and uref <= 21.6:
        znott = p20 + uref * (p21 + uref * (p22 + uref * (p23
                    + uref * (p24 + uref * p25))))
    elif uref > 21.6 and uref <= 42.6:
        znott = p30 + uref * (p31 + uref * (p32 + uref * (p33
                    + uref * (p34 + uref * p35))))
    elif uref > 42.6 and uref <= 53.0:
        znott = p40 + uref * (p41 + uref * (p42 + uref * (p43
                   + uref * (p44 + uref * p45))))
    elif uref > 53.0 and uref <= 80.0:
        znott = p50 + uref * (p51 + uref * (p52 + uref * (p53
                    + uref * (p54 + uref * (p55 + uref * p56)))))
    elif uref > 80.0:
        znott = p60
    else:
           print('Wrong input uref value:',uref)
    
    return znott

def sfc_diff(im, t1, q1, z1, wind,
             prsl1, prslki,
             sigmaf, vegtype, shdmax, ivegsrc,
             z0pert, ztpert,
             flag_iter, redrag,
             u10m, v10m, sfc_z0_type,
             wet, dry, icy,
             tskin, tsurf, snwdph, z0rl, ustar,
             cm, ch, rb, stress, fm, fh, fm10, fh2):

    for i in range(im):
        if flag_iter[i]:
            virtfac = 1.0 + RVRDM1 * max(q1[i],1.0e-8)
            thv1 = t1[i] * prslki[i] * virtfac

            if dry[i]:
                tvs = 0.5 * (tsurf[i,0] + tskin[i,0]) * virtfac
                z0max = max(1.0e-6, min(0.01 * z0rl[i,0], z1[i]))

                tem1 = 1.0 - shdmax[i]
                tem2 = tem1 * tem1
                tem1 = 1.0 - tem2

                if ivegsrc == 1:
                    if vegtype[i] == 10:
                        z0max = math.exp(tem2 * LOG01 + tem1 * LOG07)
                    elif vegtype[i] == 6:
                        z0max = math.exp(tem2 * LOG01 + tem1 * LOG05)
                    elif vegtype[i] == 7:
                        z0max = 0.01
                    elif vegtype[i] == 16:
                        z0max = 0.01
                    else:
                        z0max = math.exp(tem2 * LOG01 + tem1 * math.log(z0max))
                elif ivegsrc == 2:
                    if vegtype[i] == 7:
                        z0max = math.exp(tem2*LOG01 + tem1 * LOG07)
                    elif vegtype[i] == 8:
                        z0max = math.exp(tem2*LOG01 + tem1 * LOG05)
                    elif vegtype[i] == 9:
                        z0max = 0.01
                    elif vegtype[i] == 11:
                        z0max = 0.01
                    else:
                        z0max = math.exp(tem2 * LOG01 + tem1 + math.log(z0max))

                if z0pert[i] != 0.0:
                    z0max = z0max * (10.0 ** z0pert[i])

                z0max = max(z0max, 1.0e-6)

                czilc = 0.8

                tem1 = 1.0 - sigmaf[i]
                ztmax = z0max*math.exp(-tem1*tem1 * czilc*CA*math.sqrt(ustar[i,0] * (0.01/1.5e-5)))

                if ztpert[i] != 0.0:
                    ztmax = ztmax * (10.0 ** ztpert[i])

                ztmax = max(ztmax,1.0e-6)

                rb[i,0], fm[i,0], fh[i,0], fm10[i,0], fh2[i,0], \
                cm[i,0], ch[i,0], stress[i,0], ustar[i,0] = \
                    stability(z1[i], snwdph[i,0], thv1, wind[i], z0max, ztmax, tvs,
                              rb[i,0], fm[i,0], fh[i,0], fm10[i,0], fh2[i,0],
                              cm[i,0], ch[i,0], stress[i,0], ustar[i,0])
            
            if icy[i]:
                tvs = 0.5 * (tsurf[i,1] + tskin[i,1]) * virtfac
                z0max = max(1.0e-6, min(0.01 * z0rl[i,1], z1[i]))
                
                tem1 = 1.0 - shdmax[i]
                tem2 = tem1*tem1
                tem1 = 1.0 - tem2

                if ivegsrc == 1:
                    z0max = math.exp(tem2 * LOG01 + tem1*math.log(z0max))
                elif ivegsrc == 2:
                    z0max = math.exp(tem2 * LOG01 + tem1 * math.log(z0max))

                z0max = max(z0max, 1.0e-6)

                czilc = 0.8

                tem1 = 1.0 - sigmaf[i]
                ztmax = z0max * math.exp(-tem1*tem1*czilc*CA*math.sqrt(ustar[i,1]*(0.01/1.5e-5)))
                ztmax = max(ztmax, 1.0e-6)

                rb[i,1], fm[i,1], fh[i,1], fm10[i,1], fh2[i,1], \
                cm[i,1], ch[i,1], stress[i,1], ustar[i,1] = \
                    stability(z1[i], snwdph[i,1], thv1, wind[i], z0max, ztmax, tvs,
                            rb[i,1], fm[i,1], fh[i,1], fm10[i,1], fh2[i,1],
                            cm[i,1], ch[i,1], stress[i,1], ustar[i,1])

            if wet[i]:
                tvs = 0.5 * (tsurf[i,2] + tskin[i,2]) * virtfac
                z0 = 0.01 * z0rl[i,2]
                z0max = max(1.0e-6, min(z0, z1[i]))
                ustar[i,2] = math.sqrt(GRAV * z0 / CHARNOCK)
                wind10m = math.sqrt(u10m[i] * u10m[i] + v10m[i] * v10m[i])

                restar = max(ustar[i,2] * z0max * VISI, 0.000001)

                rat = min(7.0, 2.67 * math.sqrt(math.sqrt(restar)) - 2.57)
                ztmax = max(z0max * math.exp(-rat), 1.0e-6)

                if sfc_z0_type == 6:
                    ztmax = znot_t_v6(wind10m)
                elif sfc_z0_type == 7:
                    ztmax = znot_t_v7(wind10m)
                elif sfc_z0_type != 0:
                    print("No option for zfc_zo_type=", sfc_z0_type)
                    exit(1)
                
                rb[i,2], fm[i,2], fh[i,2], fm10[i,2], fh2[i,2], \
                cm[i,2], ch[i,2], stress[i,2], ustar[i,2] = \
                    stability(z1[i], snwdph[i,2], thv1, wind[i], z0max, ztmax, tvs,
                            rb[i,2], fm[i,2], fh[i,2], fm10[i,2], fh2[i,2],
                            cm[i,2], ch[i,2], stress[i,2], ustar[i,2])

                if sfc_z0_type == 0:
                    z0 = (CHARNOCK / GRAV) * ustar[i,2] * ustar[i,2]

                    if redrag:
                        z0rl[i,2] = 100.0 * max(min(z0, Z0S_MAX), 1.0e-7)
                    else:
                        z0rl[i,2] = 100.0 * max(min(z0, 0.1), 1.0e-7)

                elif sfc_z0_type == 6:
                    z0 = znot_m_v6(wind10m)
                    z0rl[i,2] = 100.0 * z0
                
                elif sfc_z0_type == 7:
                    z0 = znot_m_v7(wind10m)
                    z0rl[i,2] = 100.0 * z0

                else:
                    z0rl[i,2] = 1.0e-4

    return z0rl, ustar, cm, ch, rb, stress, fm, fh, fm10, fh2
