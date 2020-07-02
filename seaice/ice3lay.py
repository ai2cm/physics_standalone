##/usr/bin/env python3
import numpy as np




# function ice3lay
def ice3lay(im,kmi,fice,flag,hfi,hfd, sneti, focn, delt, lprnt, ipr, \
            snowd, hice, stsice, tice, snof):
    # constant parameters
    ds   = 330.0    # snow (ov sea ice) density (kg/m^3)
    dw   =1000.0   # fresh water density  (kg/m^3)
    dsdw = ds/dw
    dwds = dw/ds
    ks   = 0.31    # conductivity of snow   (w/mk)
    i0   = 0.3      # ice surface penetrating solar fraction
    ki   = 2.03    # conductivity of ice  (w/mk)
    di   = 917.0   # density of ice   (kg/m^3)
    didw = di/dw
    dsdi = ds/di
    ci   = 2054.0  # heat capacity of fresh ice (j/kg/k)
    li   = 3.34e5     # latent heat of fusion (j/kg-ice)
    si   = 1.0      # salinity of sea ice
    mu   = 0.054   # relates freezing temp to salinity
    tfi  = -mu*si     # sea ice freezing temp = -mu*salinity
    tfw  = -1.8     # tfw - seawater freezing temp (c)
    tfi0 = tfi-0.0001
    dici = di*ci
    dili = di*li
    dsli = ds*li
    ki4  = ki*4.0
    zero = 0.0
    one  = 1.0
    
    # vecotorize constants
    ip = np.zeros(im)
    tsf = np.zeros(im)
    ai = np.zeros(im)
    k12 = np.zeros(im)
    k32 = np.zeros(im)
    a1 = np.zeros(im)
    b1 = np.zeros(im)
    c1 = np.zeros(im)
    tmelt = np.zeros(im)
    h1 = np.zeros(im)
    h2 = np.zeros(im)
    bmelt = np.zeros(im)
    dh = np.zeros(im)
    f1 = np.zeros(im)
    #tfi = np.zeros(im)



    dt2  = 2. * delt
    dt4  = 4. * delt
    dt6  = 6. * delt
    dt2i = one / dt2

    snowd[flag] = snowd[flag]  * dwds
    hdi[flag] = (dsdw*snowd[flag] + didw * hice[flag])
    
    snowd[flag][hice[flag] < hdi] = snowd[flag][hice[flag]  < hdi] + \
        hice[flag][hice[flag] < hdi] - hdi
    hice[flag][hice[flag]  < hdi]  = hice[flag][hice[flag] < hdi] + \
            (hdi -  hice[flag][hice[flag]  < hdi]) * dsdi   
    
    snof[flag] = snof[flag] * dwds
    tice[flag] = tice[flag] -t0c
    stsice[flag,0] =  min(stsice[flag,0]-t0c, tfi0)     # degc
    stsice[flag,1] = min(stsice[flag,1]-t0c, tfi0)      # degc
    
    ip[flag] = i0 * sneti[flag] # ip +v here (in winton ip=-i0*sneti)


    tsf[flag][snowd[flag] > zero] = zero
    ip[flag][snowd[flag] > zero]  = zero

    tsf[flag][snowd[flag] <= zero] = tfi
    ip[flag][snowd[flag] <= zero] = i0 * sneti[flag][snowd[flag] <= \
            zero]  # ip +v here (in winton ip=-i0*sneti)
 

    tice[flag] = min(tice[flag], tsf[flag])    
    
    # compute ice temperature

    ai[flag]   = hfi[flag] - sneti[flag] + ip[flag] - tice[flag] e*hfd[flag]
  # +v sol input here
    k12[flag]  = ki4*ks / (ks*hice[flag] + ki4*snowd[flag])
    k32[flag]  = (ki+ki) / hice[flag]

    
    a1[flag]    = dici*hice[flag]*dt2i + k32[flag]*(dt4*k32[flag] + \
            dici*hice[flag])*one / (dt6*k32[flag] + dici*hice[flag])\
            + hfd[flag] * k12[flag] / (k12[flag] + hfd[flag])

    b1[flag]    = -di*hice[flag] * (ci*stsice[flag,0] + li* \
            tfi/stsice[flag,0]) * dt2i - ip[flag] - k32[flag]* \
            (dt4*k32[flag]*tfw + dici * hice[flag]*stsice[flag,1])* \
            one / (dt6*k32[flag] + dici * hice[flag]) + ai[flag] * \
            k12[flag] / (k12[flag] + hfd[flag])

    c1[flag]   = dili * tfi * dt2i * hice[flag]

    stsice[flag,0] = -(sqrt(b1[flag]*b1[flag] - 4.0d0*a1[flag] * \
            c1[flag]) + b1[flag])/(a1[flag]+a1[flag])
    tice[flag = (k12[flag]*stsice[flag,0] - ai[flag]) / (k12[flag] + \
            hfd[flag])

  
          if (tice(i) > tsf[flag]) then
    a1[flag][tice[flag]>tsf[flag]] = dici * \
            hice[flag][tice[flag]>tsf[flag]]*dt2i + \
            k32[flag][tice[flag]>tsf[flag]] * \ 
            (dt4*k32[flag][tice[flag]>tsf[flag]] + \
            dici*hice[flag][tice[flag]>tsf[flag]])*one / \
            (dt6*k32[flag] + dici*hice[flag][tice[flag]>tsf[flag]]) \
            + k12[flag][tice[flag]>tsf[flag]]
    b1[flag][tice[flag]>tsf[flag]] = -di * \
            hice[flag][tice[flag]>tsf[flag]] * \
            (ci*stsice[flag,0][tice[flag]>tsf[flag]] + li*tfi \
            / stsice[flag,0][tice[flag]>tsf[flag]])* dt2i - ip[flag] \
            - k32[flag][tice[flag]>tsf[flag]] * \
            (dt4*k32[flag][tice[flag]>tsf[flag]]*tfw + dici * \
            hice[flag][tice[flag]>tsf[flag]] * \
            stsice[flag,1][tice[flag]>tsf[flag]]) * one / \
            (dt6*k32[flag][tice[flag]>tsf[flag]] + dici \
            *hice[flag][tice[flag]>tsf[flag]]) - \
            k12[flag][tice[flag]>tsf[flag]] * \
            tsf[flag][tice[flag]>tsf[flag]]
    stsice[flag,0][tice[flag]>tsf[flag]] = \
            -(sqrt(b1[flag][tice[flag]>tsf[flag]] * \
            b1[flag][tice[flag]>tsf[flag]] -\
            4.0*a1[flag][tice[flag]>tsf[flag]] *\
            c1[flag][tice[flag]>tsf[flag]]) + \
            b1[flag][tice[flag]>tsf[flag]])/ \
            (a1[flag][tice[flag]>tsf[flag]] + \
            a1[flag][tice[flag]>tsf[flag]])
    tice[flag][tice[flag]>tsf[flag]] = tsf[flag][tice[flag]>tsf[flag]]
    tmelt[flag][tice[flag]>tsf[flag]] = \
            (k12[flag][tice[flag]>tsf[flag]] * \
            (stsice[flag,0][tice[flag]>tsf[flag]] - \
            tsf[flag][tice[flag]>tsf[flag]]) - \
            (ai[flag][tice[flag]>tsf[flag]] + \
            hfd[flag][tice[flag]>tsf[flag]] *\
            tsf[flag][tice[flag]>tsf[flag]])) * \
            delt[flag][tice[flag]>tsf[flag]]
              
    tmelt[flag][tice[flag]<=tsf[flag]] = zero
    snowd[flag][tice[flag]<=tsf[flag]] = \
            snow[flag][tice[flag]<=tsf[flag]]d + \
            snof[flag][tice[flag]<=tsf[flag]] * \
            delt[flag][tice[flag]<=tsf[flag]]
          
 

    stsice[flag,1] = (dt2*k32[flag]*(stsice[flag,0] + tfw + tfw) \
            +  dici*hice[flag]*stsice[flag,1]) * one / \
            (dt6*k32[flag] + dici*hice[flag])


    bmelt[flag] = (focn[flag] + \
            ki4*(stsice[flag,1] - tfw)/hice[flag]) * delt[flag]

#  --- ...  resize the ice ...

    h1[flag] = 0.5 * hice[flag]
    h2[flag] = 0.5 * hice[flag]


#  --- ...  top ...
                      
    snowmt[flag][tmelt[flag]<=snowd[flag]*dsli] = \
            tmelt[flag][tmelt[flag]<=snowd[flag]*dsli]  / dsli
    snowd[flag][tmelt[flag]<=snowd[flag]*dsli] = \
            snowd[flag][tmelt[flag]<=snowd[flag]*dsli] -\
            snowmt[flag][tmelt[flag]<=snowd[flag]*dsli]
          
    snowmt[flag][tmelt[flag]>snowd[flag]*dsli] = \
             snowd[flag][tmelt[flag]>snowd[flag]*dsli]
    h1[flag][tmelt[flag]>snowd[flag]*dsli] = \
            h1[flag][tmelt[flag]>snowd[flag]*dsli] - \
            (tmelt[flag][tmelt[flag]>snowd[flag]*dsli] - \
            snowd[tmelt[flag]>snowd[flag]*dsli]*dsli) / \
            (di * (ci - li/ \
            stsice[flag,0][tmelt[flag]>snowd[flag]*dsli]) *\
            (tfi - stsice[flag,0][tmelt[flag]>snowd[flag]*dsli]))
    snowd[tmelt[flag]>snowd[flag]*dsli] = zero
        

#  --- ...  and bottom


    dh[flag][bmelt[flag] < zero] = -bmelt[flag][bmelt[flag] < zero] \
            / (dili + dici*(tfi - tfw))
    stsice[flag,1][bmelt[flag] < zero]=(h2[flag][bmelt[flag] < zero]\
            *stsice[flag,1][bmelt[flag] < zero] + \
            dh[flag][bmelt[flag] < zero]*tfw) / \
            (h2[flag][bmelt[flag] < zero] + \
            dh[flag][bmelt[flag] < zero])
    h2[flag][bmelt[flag] < zero] = h2[flag][bmelt[flag] < zero] + \
            dh[flag][bmelt[flag] < zero]
    
    h2[flag][bmelt[flag] <= zero] = h2[flag][bmelt[flag] <= zero] - \
            bmelt[flag][bmelt[flag] <= zero] / \
            (dili + dici*(tfi - stsice[flag,1][bmelt[flag] <= zero]))
          

#  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow

    hice[flag] = h1[flag] + h2[flag]

          
    # begin if_hice_block
    # begin if_h1_block

    f1[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]=one-\
            (h2[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]+\
            h2[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]])\
            / hice[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]
    stsice[flag,1][hice[flag]>zero][h1[flag]>0.5*hice[flag]] = \
            f1[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]] * \
            (stsice[flag,0][hice[flag]>zero][h1[flag]>0.5*hice[flag]]\
            + li*tfi/ \
            (ci* \
            stsice[flag,0][hice[flag]>zero][h1[flag]>0.5*hice[flag]]))\
            + (one - \
            f1[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]) * \
            stsice[flag,1][hice[flag]>zero][h1[flag]>0.5*hice[flag]]

    # begin if_stsice_block

    hice[flag][hice[flag]>zero][h1[flag]>0.5*hice[flag]]\
            [stsice[flag,1]>tfi]=hice[flag][hice[flag]>zero]\
            [h1[flag]>0.5*hice[flag]][stsice[flag,1]>tfi]\
            -h2[flag][hice[flag]>zero][h1[flag]>0.5*\
            hice[flag]][stsice[flag,1]>tfi]*ci*(stsice[flag,1]\
            [hice[flag]>zero][h1[flag]>0.5*hice[flag]]\
            [stsice[flag,1]>tfi] - tfi)/(li*delt[hice[flag]>zero]\
            [h1[flag]>0.5*hice[flag]][stsice[flag,1]>tfi])

    stsice[flag,1][hice[flag]>zero][h1[flag]>0.5*hice[flag]]\
            [stsice[flag,1]>tfi] = tfi
              
    # end if_stsice_block

    # else if_h1_block
              
    f1[flag][flag,1][hice[flag]>zero][h1[flag]>0.5*hice[flag]] = \
            (h1[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]+\
            h1[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]) / \
            hice[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]

    stsice[flag,0][hice[flag]>zero][h1[flag]<=0.5*hice[flag]] = \
            f1[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]*(\
            stsice[flag,0][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]\
            + li*tfi/(ci*stsice[flag,0][hice[flag]>zero]\
            [h1[flag]<=0.5*hice[flag]]))+(one - \
            f1[flag][hice[flag]>zero][h1[flag]<=0.5*hice[flag]])\
            *stsice[flag,1][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]

    stsice[flag,0][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]= (\
            stsice[flag,0][hice[flag]>zero][h1[flag]<=0.5*hice[flag]]\
            - sqrt(stsice[flag,0][hice[flag]>zero]\
            [h1[flag]<=0.5*hice[flag]]*stsice[flag,0]\
            [hice[flag]>zero][h1[flag]<=0.5*hice[flag]] \
            - 4.0*tfi*li/ci)) * 0.5

    # end if_h1_block

    k12[flag][hice[flag]>zero] = ki4*ks / (ks* \
            hice[flag][hice[flag]>zero] + ki4* \
            snowd[flag][hice[flag]>zero])

    gflux[flag][hice[flag]>zero] = k12[flag][hice[flag]>zero] * \
            (stsice[flag,0][hice[flag]>zero] -\
            tice[flag][hice[flag]>zero])
    
    # else if_hice_block

    snowd[flag][hice[flag]<=zero] = snowd[flag][hice[flag]<=zero] + \
            (h1[flag][hice[flag]<=zero]*(ci*\
            (stsice[flag,0][hice[flag]<=zero] - tfi)- li*(one - tfi/ \
            stsice[flag,0][hice[flag]<=zero])) +\
            h2[flag][hice[flag]<=zero]*(ci*\
            (stsice[flag,1][hice[flag]<=zero] - tfi) - li)) / li

    hice[flag][hice[flag]<=zero] = max(zero, \
            snowd[flag][hice[flag]<=zero]*dsdi)

    snowd[flag][hice[flag]<=zero] = zero

    stsice[flag,0][hice[flag]<=zero] = tfw

    stsice[flag,1][hice[flag]<=zero] = tfw

    gflux[flag][hice[flag]<=zero]    = zero
    
    # end if_hice_block

    gflux[flag] = fice[flag] * gflux[flag]
    snowmt[flag] = snowmt[flag] * dsdw
    snowd[flag] = snowd[flag] * dsdw
    tice[flag]  = tice[flag]     + t0c
    stsice(i,1) = stsice(i,1) + t0c
    stsice(i,2) = stsice(i,2) + t0c
    
    # end if_flag_block


    return snowd, hice, stsice, tice, snof, snowmt, gflux
