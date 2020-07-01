#!/usr/bin/env python3


# python function sfc sea ice


#  --- ...  set flag for sea-ice

    flag = (islimsk == 2) and flag_iter
    hice[flag_iter and (islimsk < 2)] = zero
    fice[flag_iter and (islimsk < 2)] = zero

# TODO: save mask "flag and srflag > zero" as logical array?
    ep[flag and srflag > zero]    = ep[flag and srflag > zero]*(one-srflag[flag and srflag > zero])
    weasd[flag and srflag > zero] = weasd[flag and srflag > zero] + 1.e3*tprcp[flag and srflag > zero]*srflag[flag and srflag > zero]
    tprcp[flag and srflag > zero] = tprcp[flag and srflag > zero]*(one-srflag[flag and srflag > zero])
#  --- ...  update sea ice temperature

    stsice[flag, :] = stci[flag, :]

#  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
#           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
#           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
#           is sat. hum. at surface
#           convert slrad to the civilized unit from langley minute-1 k-4

#         dlwflx has been given a negative sign for downward longwave
#         sfcnsw is the net shortwave flux (direction: dn-up)

    q0[flag]  = np.max(q1[flag], 1.0e-8)
    theta1[flag] = t1[flag] * prslki[flag]
    rho[flag]    = prsl1[flag] / (rd*t1[flag]*(one+rvrdm1*q0))
    qs1       = fpvs(t1[flag])
    qs1       = np.max(eps*qs1 / (prsl1[flag] + epsm1*qs1), 1.e-8)
    q0        = min(qs1, q0)
  
    if any(fice[flag] < cimin):
        print("warning: ice fraction is low:", fice[flag][fice[flag] < cimin])
        fice[flag][fice[flag] < cimin] = cimin
        tice[flag][fice[flag] < cimin] = tgice
        tskin[flag][fice[flag] < cimin]= tgice
        print('fix ice fraction: reset it to:', fice[flag][fice[flag] < cimin])
  
    ffw[flag]    = 1.0 - fice[flag]
  
    qssi = fpvs(tice[flag])
    qssi = eps*qssi / (ps[flag] + epsm1*qssi)
    qssw = fpvs(tgice)
    qssw = eps*qssw / (ps[flag] + epsm1*qssw)

#  --- ...  snow depth in water equivalent is converted from mm to m unit

    snowd[flag] = weasd[flag] * 0.001

#  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
#           soil is allowed to interact with the atmosphere.
#           we should eventually move to a linear combination of soil and
#           snow under the condition of patchy snow.

#  --- ...  rcp = rho cp ch v

    cmm[flag] = cm[flag]  * wind[flag]
    chh[flag] = rho[flag] * ch[flag] * wind[flag]
    rch[flag] = chh[flag] * cp

#  --- ...  sensible and latent heat flux over open water & sea ice

    evapi[flag] = elocp * rch[flag] * (qssi - q0)
    evapw[flag] = elocp * rch[flag] * (qssw - q0)

    snetw[flag] = sfcdsw[flag] * (one - albfw)
    snetw[flag] = np.min(3.*sfcnsw[flag]/(one+2.*ffw[flag]), snetw[flag])
    sneti[flag] = (sfcnsw[flag] - ffw[flag]*snetw[flag]) / fice[flag]

    t12 = tice[flag] * tice[flag]
    t14 = t12 * t12

#  --- ...  hfi = net non-solar and upir heat flux @ ice surface

    hfi[flag] = -dlwflx[flag] + sfcemis[flag]*sbc*t14 + evapi[flag]           /
          + rch[flag]*(tice[flag] - theta1[flag])
    hfd[flag] = 4.*sfcemis[flag]*sbc*tice[flag]*t12                     /
          + (one + elocp*eps*hvap*qs1/(rd*t12)) * rch[flag]


    t12 = tgice * tgice
    t14 = t12 * t12

#  --- ...  hfw = net heat flux @ water surface (within ice)

    focn[flag] = 2.   # heat flux from ocean - should be from ocn model
    snof[flag] = zero    # snowfall rate - snow accumulates in gbphys
  
    hice[flag] = np.max( np.min( hice[flag], himax ), himin )
    snowd[flag] = np.min( snowd[flag], hsmax )

# TODO: write more efficiently, save mask as new variable?
    if any(snowd[flag] > (2.*hice[flag])):
        print('warning: too much snow :', snowd[flag][snowd[flag] > (2.*hice[flag])])
        snowd[flag][snowd[flag] > (2.*hice[flag])] = hice[flag][snowd[flag] > (2.*hice[flag])] + hice[flag][snowd[flag] > (2.*hice[flag])]
        print('fix: decrease snow depth to:',snowd[flag][snowd[flag] > (2.*hice[flag])])

# call function ice3lay
    snowd, hice, stsice, tice, snof, snowmt, gflux = ice3lay(                                                      
           im, kmi, fice, flag, hfi, hfd, sneti, focn, delt, lprnt, ipr)

      do i = 1, im
        if (flag(i)) then
          if any(tice[flag] < timin):
            print('warning: snow/ice temperature is too low:',
                  tice[flag][tice[flag] < timin], ' i=', i)
            tice[flag] = timin
            print *,'fix snow/ice temperature: reset it to:',tice[flag]
          endif

          if (stsice(i,1) < timin) then
            print *,'warning: layer 1 ice temp is too low:',stsice(i,1) &
     &,             ' i=',i
            stsice(i,1) = timin
            print *,'fix layer 1 ice temp: reset it to:',stsice(i,1)
          endif

          if (stsice(i,2) < timin) then
            print *,'warning: layer 2 ice temp is too low:',stsice(i,2)
            stsice(i,2) = timin
            print *,'fix layer 2 ice temp: reset it to:',stsice(i,2)
          endif

          tskin[flag] = tice[flag]*fice[flag] + tgice*ffw[flag]
        endif
      enddo


# edited until here

    stc[flag,:] = np.min(stsice[flag,k], t0c)

      do i = 1, im
        if (flag(i)) then
#  --- ...  calculate sensible heat flux (& evap over sea ice)

          hflxi    = rch(i) * (tice(i) - theta1(i))
          hflxw    = rch(i) * (tgice - theta1(i))
          hflx(i)  = fice(i)*hflxi    + ffw(i)*hflxw
          evap(i)  = fice(i)*evapi(i) + ffw(i)*evapw(i)
#
#  --- ...  the rest of the output

          qsurf(i) = q1(i) + evap(i) / (elocp*rch(i))

#  --- ...  convert snow depth back to mm of water equivalent

          weasd(i)  = snowd(i) * 1000.0
          snwdph(i) = weasd(i) * dsi             # snow depth in mm

          tem     = 1.0 / rho(i)
          hflx(i) = hflx(i) * tem * cpinv
          evap(i) = evap(i) * tem * hvapi
        endif
      enddo
#
