from types import DynamicClassAttribute
import numpy as np
import xarray as xr
import os
import sys

from gt4py.gtscript import (
    stencil,
	function,
    computation,
    interval,
    __externals__,
    PARALLEL,
    FORWARD,
    BACKWARD,
    min,
    max,
	cos,
	log,
)

from config import *
from radphysparam import *
from phys_const import con_eps, con_epsm1, con_rocp, \
                       con_fvirt, con_rog, con_epsq, \
					   con_ttp, con_pi
from funcphys_gt import fpvs

backend = "gtc:gt:cpu_ifirst"

# convert pressure unit from pa to mb
@stencil(backend=backend, 
        externals={"con_eps" : con_eps,
                   "con_epsm1" : con_epsm1,
				   "con_rocp" : con_rocp,
                  }
)
def pressure_convert(plvl : FIELD_FLT,
                     prsi : FIELD_FLT,
                     plyr : FIELD_FLT,
                     prsl : FIELD_FLT,
                     tlyr : FIELD_FLT,
                     tgrs : FIELD_FLT,
                     prslk1 : FIELD_FLT,
                     prslk  : FIELD_FLT,
                     rhly : FIELD_FLT,
                     qgrs : Field[(DTYPE_FLT,(8,))],
                     qstl : FIELD_FLT,
                     tracer1 : Field[(DTYPE_FLT,(8,))],
                     QMIN : DTYPE_FLT,
                     NTRAC : DTYPE_INT,
                     ivflip : int,
                     lsk : int,
					 ):
	from __externals__ import (con_eps, con_epsm1, con_rocp)
	with computation(PARALLEL), interval(0,-1):
		plvl = prsi[0,0,0] * 0.01
	with computation(PARALLEL), interval(1,None):
		plyr = prsl[0,0,0] * 0.01
		tlyr = tgrs[0,0,0]
		prslk1 = prslk

		es = min(prsl[0,0,0], fpvs(tgrs[0,0,0]))
		qs = max(QMIN, con_eps * es / (prsl[0,0,0] + con_epsm1 * es))
		rhly = max(0.0, min(1.0, max(QMIN, qgrs[0,0,0][0])/qs))
		qstl = qs
		for i in range(NTRAC):
			tracer1[0,0,0][i] = max(0.0, qgrs[0,0,0][i])
	with computation(PARALLEL), interval(1,2):
		if ivflip == 0:
			plvl = 0.01 * prsi[0,0,-1]
			if lsk != 0:
				plvl = 0.5 * (plvl[0,0,1] + plvl[0,0,0])
	with computation(PARALLEL), interval(-1,None):
		if ivflip != 0:
			plvl = 0.01 * prsi[0,0,0]
	with computation(PARALLEL), interval(-2,-1):
		if ivflip != 0 and lsk != 0:
			plvl = 0.5 * (plvl[0,0,1] + plvl[0,0,0])

# Values for extra top layer
@stencil(backend=backend, 
        externals={
				   "con_rocp" : con_rocp,
                  }
)
def extra_values(plvl : FIELD_FLT,
				 prsi : FIELD_FLT,
				 plyr : FIELD_FLT,
				 prsl : FIELD_FLT,
				 tlyr : FIELD_FLT,
				 prslk1 : FIELD_FLT,
				 rhly : FIELD_FLT,
				 qstl : FIELD_FLT,
				 tracer1 : Field[(DTYPE_FLT,(8,))],
				 prsmin : float,
				 lla : float,
				 llb : float,
				 ):
	from __externals__ import (con_rocp)
		# *** Values for extra top layer ***
	with computation(PARALLEL), interval(0,1):
		if lla == 2 and llb == 1:
			plvl = prsmin
	
	with computation(FORWARD), interval(1,2):
		if lla == 2 and llb == 1:
			if plvl[0,0,0] <= prsmin:
				plvl = 2.0 * prsmin
	
			plyr = 0.5 * plvl[0,0,0]
			tlyr = tlyr[0,0,1]
			prslk1 = (plyr[0,0,0] * 0.0001) ** con_rocp
			rhly = rhly[0,0,1]
			qstl = qstl[0,0,1]
			for i in range(8):
				tracer1[0,0,0][i] = tracer1[0,0,1][i]

	with computation(PARALLEL), interval(-1,None):
		if lla == 63 and llb == 64:
			plvl = prsmin
	
	with computation(PARALLEL), interval(-2,-1):
		if lla == 63 and llb == 64:
			if plvl[0,0,0] <= prsmin:
				plvl = 2.0 * prsmin

	with computation(FORWARD), interval(-1,None):
		if lla == 63 and llb == 64:
			plyr = 0.5 * plvl[0,0,-2]
			tlyr = tlyr[0,0,-1]
			prslk1 = (plyr[0,0,0] * 0.0001) ** con_rocp
			rhly = rhly[0,0,-1]
			qstl = qstl[0,0,-1]	
			for j in range(8):
				tracer1[0,0,0][j] = tracer1[0,0,-1][j]

@stencil(backend=backend)
def getozn(olyr : FIELD_FLT,
		   tracer1 : Field[(DTYPE_FLT,(8,))],
		   QMIN : float,
		   ntoz : DTYPE_INT,
          ):
	with computation(PARALLEL), interval(1,None):
		olyr[0,0,0] = max(QMIN, tracer1[0,0,0][ntoz-1])

@stencil(backend=backend)
def coszmn_stencil_1(coslat : FIELD_1D,
					 coszen : FIELD_2D,
					 istsun : FIELD_2D,
					 sinlat : FIELD_1D,
					 xlon   : FIELD_1D,
					 cdec   : float,
					 cns    : float,
					 czlimit: float,
					 sdec   : float):
	with computation(FORWARD), interval(0,1):
		coszn = sdec * sinlat[0] + cdec * coslat[0] * cos(
                    cns + xlon[0]
                )
		coszen[0,0] = coszen[0,0] + max(0.0, coszn)
		if coszn > czlimit:
			istsun[0,0] = istsun[0,0] + 1

@stencil(backend=backend)
def coszmn_stencil_2(coszdg : FIELD_2D,
					 coszen : FIELD_2D,
					 istsun : FIELD_2D,
					 rstp   : float):
	with computation(FORWARD), interval(0,1):
		coszdg[0,0] = coszen[0,0] * rstp
		if istsun[0,0] > 0:
			coszen[0,0] = coszen[0,0] / istsun[0,0]

@stencil(backend=backend)
def getgases_stencil(gasdat : Field[type_10],
					 co2vmr : float,
					 n2ovmr : float,
					 ch4vmr : float,
					 o2vmr  : float,
					 covmr  : float,
					 f11vmr : float,
					 f12vmr : float,
					 f22vmr : float,
					 cl4vmr : float,
					 f113vmr: float,
					#  ico2flg: int,
					#  gco2cyc: float,
					#  co2_glb: float,
					#  raddeg : float,
					#  resco2 : float,
					 ):
	with computation(FORWARD), interval(1,None):
		gasdat[0,0,0][0] = co2vmr
		gasdat[0,0,0][1] = n2ovmr
		gasdat[0,0,0][2] = ch4vmr
		gasdat[0,0,0][3] = o2vmr
		gasdat[0,0,0][4] = covmr
		gasdat[0,0,0][5] = f11vmr
		gasdat[0,0,0][6] = f12vmr
		gasdat[0,0,0][7] = f22vmr
		gasdat[0,0,0][8] = cl4vmr
		gasdat[0,0,0][9] = f113vmr 

@stencil(backend=backend,
		 externals={"con_fvirt" : con_fvirt,
		 			"con_rog" : con_rog,
		  		   }
		)
def get_layer_temp(tem2da : FIELD_FLT,
				   tem2db : FIELD_FLT,
				   plyr   : FIELD_FLT,
				   plvl   : FIELD_FLT,
				   tlyr   : FIELD_FLT,
				   tlvl   : FIELD_FLT,
				   tgrs   : FIELD_FLT,
				   qlyr   : FIELD_FLT,
				   delp   : FIELD_FLT,
				   tvly   : FIELD_FLT,
				   tem1d  : FIELD_2D,
				   tsfa   : FIELD_2D,
				   tskn   : FIELD_2D,
				   qgrs   : Field[(DTYPE_FLT,(8,))],
				   ivflip : int,
				   prsmin : float,
				   QME5   : float,
				   QME6   : float,
				   lextop : bool,
				   ):
	from __externals__ import (con_fvirt, con_rog)
	with computation(PARALLEL), interval(1,None):
		tem2da = log(plyr[0,0,0])
	with computation(PARALLEL), interval(0,-1):
		tem2db = log(plvl[0,0,0])
	
	with computation(FORWARD), interval(0,1):
		if ivflip == 0:
			tem2db = log(max(prsmin, plvl[0,0,0]))
			tlvl   = tlyr[0,0,1]
			tem1d = QME6
		else:
			tem1d = QME6
			tem2db = log(plvl[0,0,0])
			tsfa = tlyr[0,0,1]
			tlvl = tskn[0,0]

	with computation(FORWARD), interval(-1,None):
		if ivflip == 0:
			tsfa = tlyr[0,0,0]
			tlvl = tskn[0,0]
		else:
			tem2db = log(max(prsmin,plvl[0,0,0]))
			tlvl = tlyr[0,0,0]
	
	with computation(FORWARD), interval(1,None):
		if ivflip == 0:
			qlyr = max(tem1d[0,0], qgrs[0,0,0][0])
			tem1d = min(QME5, qlyr[0,0,0])
			tvly = tgrs[0,0,0] * (1.0 + con_fvirt * qlyr[0,0,0])
			delp = plvl[0,0,0] - plvl[0,0,-1]
	
	with computation(FORWARD), interval(1,2):
		if lextop and ivflip == 0:
			qlyr = qlyr[0,0,1]
			tvly = tvly[0,0,1]
			delp = plvl[0,0,0] - plvl[0,0,-1]

	with computation(PARALLEL), interval(1,-2):
		if ivflip == 0:
			tlvl = tlyr[0,0,1] + (tlyr[0,0,0] - tlyr[0,0,1]) * (
				tem2db[0,0,0] - tem2da[0,0,1]
			) / (tem2da[0,0,0] - tem2da[0,0,1])
			dz = 0.001*con_rog * (tem2db[0,0,0] - tem2db[0,0,-1]) * tvly[0,0,0]

	with computation(BACKWARD), interval(1,None):
		if ivflip != 0:
			qlyr = max(tem1d[0,0],qgrs[0,0,0][0])
			tem1d = min(QME5, qlyr[0,0,0])
			tvly  = tgrs[0,0,0] * (1.0 + con_fvirt * qlyr[0,0,0])
			delp  = plvl[0,0,-1] - plvl[0,0,0]

	with computation(FORWARD), interval(-1,None):
		if lextop and ivflip != 0:
			qlyr = qlyr[0,0,-1]
			tvly = tvly[0,0,-1]
			delp = plvl[0,0,-2] - plvl[0,0,-1]
	
	with computation(PARALLEL), interval(1,-1):
		if ivflip != 0:
			tlvl = tlyr[0,0,0] + (tlyr[0,0,1] - tlyr[0,0,0]) * (
				    tem2db[0,0,0] - tem2da[0,0,0]
				   )  / (tem2da[0,0,1] - tem2da[0,0,0])

			dz = 0.001 * con_rog * (tem2db[0,0,-1] - tem2db[0,0,0]) * tvly[0,0,0]

@function
def lat_lon_convert(alat, alon, xlat, xlon, rdg):
	alon = xlon * rdg
	if alon < 0.0:
		alon = alon + 360.0
	alat = xlat * rdg  # if xlat in pi/2 -> -pi/2 range
	return alat, alon

@function
def comp_height_layer_fxn(prsi, p_val, rovg, tvly):
	return rovg * (log(prsi) - log(p_val)) * tvly

@stencil(backend=backend)
def aerosol_stencil(alat : FIELD_2D,
					alon : FIELD_2D,
					xlat : FIELD_1D,
					xlon : FIELD_1D,
					rdg  : float,
					):
	with computation(FORWARD), interval(0,1):
		alat, alon = lat_lon_convert(alat, alon, xlat, xlon, rdg)

@stencil(backend=backend)
def comp_height_layer_ivflip1(dz : FIELD_FLT, 
					  hz : FIELD_FLT,
					  prsi : FIELD_FLT, 
					  prsl : FIELD_FLT, 
					  tvly : FIELD_FLT,
					  rovg : float,):
	with computation(PARALLEL):
		with interval(0,-1):
			dz = comp_height_layer_fxn(prsi, prsi[0,0,1], rovg, tvly[0,0,1])
		with interval(-1,None):
			dz = comp_height_layer_fxn(prsi, prsl[0,0,1], rovg, tvly)
			dz = 2.0 * dz[0,0,0]
	
	with computation(FORWARD):
		with interval(0,1):
			hz = 0.0
		with interval(1,None):
			hz = hz[0,0,-1] + dz[0,0,-1]

@stencil(backend=backend)
def comp_height_layer_ivflip0(dz : FIELD_FLT, 
					  hz : FIELD_FLT,
					  prsi : FIELD_FLT, 
					  prsl : FIELD_FLT, 
					  tvly : FIELD_FLT,
					  rovg : float,):
	with computation(PARALLEL):
		with interval(0,1):
			dz = comp_height_layer_fxn(prsi[0,0,1], prsl[0,0,1], rovg, tvly[0,0,1])
			dz = 2.0 * dz[0,0,0]
		with interval(1,None):
			dz = comp_height_layer_fxn(prsi[0,0,1], prsi, rovg, tvly[0,0,1])

	with computation(BACKWARD):
		with interval(-1,None):
			hz = 0.0
		with interval(0,-1):
			hz = hz[0,0,1] + dz[0,0,0]

@stencil(backend=backend)
def bound_interpol(alat : FIELD_2D,
					   volcae : FIELD_2D,
					   ivolae_0 : float,
					   ivolae_1 : float,
					   ivolae_2 : float,
					   ivolae_3 : float,
					   ):
	with computation(FORWARD), interval(0,1):
		if alat[0,0] > 46.0:
			volcae = 1.0e-4 * ivolae_0
		elif alat[0,0] > 44.0:
			volcae = 5.0e-5 * (ivolae_0 + ivolae_1)
		elif alat[0,0] > 1.0:
			volcae = 1.0e-4 * ivolae_1
		elif alat[0,0] > -1.0:
			volcae = 5.0e-5 * (ivolae_1 + ivolae_2)
		elif alat[0,0] > -44.0:
			volcae = 1.0e-4 * ivolae_2
		elif alat[0,0] > -46.0:
			volcae = 5.0e-5 * (ivolae_2 + ivolae_3)
		else:
			volcae = 1.0e-4 * ivolae_3

@stencil(backend=backend,
		 externals={
				   "con_epsq" : con_epsq,
                   }
		)

def cloud_comp_1(ccnd : Field[(DTYPE_FLT, (1,))],
				 tracer1 : Field[(DTYPE_FLT,(8,))],
				 tracer_idx_0: int,
				):
	from __externals__ import(con_epsq)
	with computation(PARALLEL), interval(1,None):
		ccnd[0,0,0][0] = tracer1[0,0,0][tracer_idx_0]
		if ccnd[0,0,0][0] < con_epsq:
			ccnd[0,0,0][0] = 0.0


@stencil(backend=backend,
		 externals={
				   "con_epsq" : con_epsq,
                   }
		)
def cloud_comp_2(ccnd : Field[(DTYPE_FLT, (2,))],
				 tracer1 : Field[(DTYPE_FLT,(8,))],
				 tracer_idx_0 : int,
				 tracer_idx_1 : int,
				):
	from __externals__ import(con_epsq)
	with computation(PARALLEL), interval(1,None):
		ccnd[0,0,0][0] = tracer1[0,0,0][tracer_idx_0]
		ccnd[0,0,0][1] = tracer1[0,0,0][tracer_idx_1]
		if ccnd[0,0,0][0] < con_epsq:
			ccnd[0,0,0][0] = 0.0
		if ccnd[0,0,0][1] < con_epsq:
			ccnd[0,0,0][1] = 0.0

@stencil(backend=backend,
		 externals={
				   "con_epsq" : con_epsq,
                   }
		 )
def cloud_comp_4(ccnd : Field[(DTYPE_FLT, (4,))],
				 tracer1 : Field[(DTYPE_FLT,(8,))],
				 tracer_idx_0 : int,
				 tracer_idx_1 : int,
				 tracer_idx_2 : int,
				 tracer_idx_3 : int,
				):
	from __externals__ import(con_epsq)
	with computation(PARALLEL), interval(1,None):
		ccnd[0,0,0][0] = tracer1[0,0,0][tracer_idx_0]
		ccnd[0,0,0][1] = tracer1[0,0,0][tracer_idx_1]
		ccnd[0,0,0][2] = tracer1[0,0,0][tracer_idx_2]
		ccnd[0,0,0][3] = tracer1[0,0,0][tracer_idx_3]
		if ccnd[0,0,0][0] < con_epsq:
			ccnd[0,0,0][0] = 0.0
		if ccnd[0,0,0][1] < con_epsq:
			ccnd[0,0,0][1] = 0.0
		if ccnd[0,0,0][2] < con_epsq:
			ccnd[0,0,0][2] = 0.0
		if ccnd[0,0,0][3] < con_epsq:
			ccnd[0,0,0][3] = 0.0

@stencil(backend=backend,
		 externals={
				   "con_epsq" : con_epsq,
                   }
		)
def cloud_comp_5(ccnd : Field[(DTYPE_FLT, (4,))],
				 tracer1 : Field[(DTYPE_FLT,(8,))],
				 tracer_idx_0 : int,
				 tracer_idx_1 : int,
				 tracer_idx_2 : int,
				 tracer_idx_3 : int,
				 tracer_idx_4 : int,
				):
	from __externals__ import(con_epsq)
	with computation(PARALLEL), interval(1,None):
		ccnd[0,0,0][0] = tracer1[0,0,0][tracer_idx_0]
		ccnd[0,0,0][1] = tracer1[0,0,0][tracer_idx_1]
		ccnd[0,0,0][2] = tracer1[0,0,0][tracer_idx_2]
		ccnd[0,0,0][3] = tracer1[0,0,0][tracer_idx_3] + tracer1[0,0,0][tracer_idx_4]

		if ccnd[0,0,0][0] < con_epsq:
			ccnd[0,0,0][0] = 0.0
		if ccnd[0,0,0][1] < con_epsq:
			ccnd[0,0,0][1] = 0.0
		if ccnd[0,0,0][2] < con_epsq:
			ccnd[0,0,0][2] = 0.0
		if ccnd[0,0,0][3] < con_epsq:
			ccnd[0,0,0][3] = 0.0

@stencil(backend=backend)
def cloud_comp_5_v2(ccnd : Field[(DTYPE_FLT, (1,))],
				 tracer1 : Field[(DTYPE_FLT,(8,))],
				 tracer_idx_0 : int,
				 tracer_idx_1 : int,
				 tracer_idx_2 : int,
				 tracer_idx_3 : int,
				 tracer_idx_4 : int,
				 EPSQ : float,
				):
	with computation(PARALLEL), interval(1,None):
		ccnd[0,0,0][0] = tracer1[0,0,0][tracer_idx_0] + tracer1[0,0,0][tracer_idx_1] + \
						 tracer1[0,0,0][tracer_idx_2] + tracer1[0,0,0][tracer_idx_3] + \
						 tracer1[0,0,0][tracer_idx_4]


@stencil(backend=backend)
def ccnd_zero(ccnd : Field[(DTYPE_FLT, (1,))],
 			  EPSQ : float,
			 ):
	with computation(PARALLEL), interval(1,None):
		if ccnd[0,0,0][0] < EPSQ:
			ccnd[0,0,0][0] = 0.0

@stencil(backend=backend)
def cloud_cover(cldcov : FIELD_FLT,
				effrl : FIELD_FLT,
				effri : FIELD_FLT,
				effrr : FIELD_FLT,
				effrs : FIELD_FLT,
				phy_f3d : Field[(DTYPE_FLT,(8,))],
				tracer1 : Field[(DTYPE_FLT,(8,))],
				indcld_idx : int, 
				ntclamt_idx : int,							
				uni_cld : bool,
				effr_in : bool,
				imp_physics : DTYPE_INT,
				):
	with computation(PARALLEL), interval(1,None):
		if uni_cld:
			cldcov = phy_f3d[0,0,0][indcld_idx]
			if effr_in:
				effrl = phy_f3d[0,0,0][1]
				effri = phy_f3d[0,0,0][2]
				effrr = phy_f3d[0,0,0][3]
				effrs = phy_f3d[0,0,0][4]

		elif imp_physics == 11:
			cldcov = tracer1[0,0,0][ntclamt_idx]
			if effr_in:
				effrl = phy_f3d[0,0,0][0]
				effri = phy_f3d[0,0,0][1]
				effrr = phy_f3d[0,0,0][2]
				effrs = phy_f3d[0,0,0][3]
		else:
			cldcov = 0.0

@stencil(backend=backend)
def add_cond_cloud_water(deltaq : FIELD_FLT,
						 cnvw   : FIELD_FLT,
						 cnvc   : FIELD_FLT,
						 cldcov : FIELD_FLT,
						 effrl : FIELD_FLT,
						 effri : FIELD_FLT,
						 effrr : FIELD_FLT,
						 effrs : FIELD_FLT,
						 phy_f3d : Field[(DTYPE_FLT,(8,))],
						 ccnd : Field[(DTYPE_FLT, (1,))],
						 num_p3d : DTYPE_INT,
						 npdf3d  : DTYPE_INT,
						 ncnvcld3d : DTYPE_INT,
						 imp_physics : DTYPE_INT,
						 lextop : bool,
						 ivflip : int,
						 effr_in : bool,
						):
		with computation(PARALLEL), interval(1,None):
			if num_p3d == 4 and npdf3d == 3:
				deltaq = phy_f3d[0,0,0][4]
				cnvw = phy_f3d[0,0,0][5]
				cnvc = phy_f3d[0,0,0][6]
			elif npdf3d == 0 and ncnvcld3d == 1:
				deltaq = 0.0
				cnvw = phy_f3d[0,0,0][num_p3d]
				cnvc = 0.0
			else:
				deltaq = 0.0
				cnvw = 0.0
				cnvc = 0.0

		with computation(PARALLEL):
			with interval(1,2):
				if lextop and ivflip != 1:
					cldcov = cldcov[0,0,1]
					deltaq = deltaq[0,0,1]
					cnvw = cnvw[0,0,1]
					cnvc = cnvc[0,0,1]
					if effr_in:
						effrl = effrl[0,0,1]
						effri = effri[0,0,1]
						effrr = effrr[0,0,1]
						effrs = effrs[0,0,1]
				if imp_physics == 99:
					ccnd[0,0,0][0] = ccnd[0,0,0][0] + cnvw[0,0,0]
			with interval(2,-1):
				if imp_physics == 99:
					ccnd[0,0,0][0] = ccnd[0,0,0][0] + cnvw[0,0,0]
			with interval(-1,None):
				if lextop and ivflip == 1:
					cldcov = cldcov[0,0,-1]
					deltaq = deltaq[0,0,-1]
					cnvw = cnvw[0,0,-1]
					cnvc = cnvc[0,0,-1]
					if effr_in:
						effrl = effrl[0,0,-1]
						effri = effri[0,0,-1]
						effrr = effrr[0,0,-1]
						effrs = effrs[0,0,-1]
				if imp_physics == 99:
					ccnd[0,0,0][0] = ccnd[0,0,0][0] + cnvw[0,0,0]

@stencil(backend=backend,
		 externals = {
			 "con_ttp" : con_ttp,
			 "con_pi" : con_pi,
		 	}
		 )
def progcld4_stencil(rew : FIELD_FLT,
					 rei : FIELD_FLT,
					 rer : FIELD_FLT,
					 res : FIELD_FLT,
					 tem2d : FIELD_FLT,
					 clwf : FIELD_FLT,
					 clw : Field[(DTYPE_FLT, (1,))],
					 ptop1 : Field[gtscript.IJ,(DTYPE_FLT,(4,))],
					 clouds : Field[(DTYPE_FLT, (9,))],
					 tlyr : FIELD_FLT,
					 rxlat : FIELD_2D,
					 xlat : FIELD_1D,
					 cnvw : FIELD_FLT,
					 delp : FIELD_FLT,
					 cip : FIELD_FLT,
					 cwp : FIELD_FLT,
					 slmsk : FIELD_1D,
					 cldtot : FIELD_FLT,
					 plyr : FIELD_FLT,
					 tvly : FIELD_FLT,
					 de_lgth : FIELD_2D,
					 gord : DTYPE_FLT,
					 climit : DTYPE_FLT,
					 climit2 : DTYPE_FLT,
					 gfac : DTYPE_FLT,
					 iovr : int, 					 
					 reliq_def : DTYPE_FLT,
					 reice_def : DTYPE_FLT,
					 rrain_def : DTYPE_FLT,
					 rsnow_def : DTYPE_FLT,
					 ptopc_0_0 : DTYPE_FLT,
					 ptopc_0_1 : DTYPE_FLT,
					 ptopc_1_0 : DTYPE_FLT,
					 ptopc_1_1 : DTYPE_FLT,
					 ptopc_2_0 : DTYPE_FLT,
					 ptopc_2_1 : DTYPE_FLT,
					 ptopc_3_0 : DTYPE_FLT,
					 ptopc_3_1 : DTYPE_FLT,
					 lcrick : bool,
					 lcnorm : bool,
					 ):
	from __externals__ import(con_ttp, con_pi)
	with computation(PARALLEL):
		with interval(0,1):
			rew = reliq_def
			rei = reice_def
			rer = rrain_def
			res = rsnow_def
			tem2d = min(1.0, max(0.0,(con_ttp - tlyr[0,0,1])*0.05))
			if lcrick: 
				clwf = 0.75* clw[0,0,1][0] + 0.25 * clw[0,0,2][0]
			else:
				clwf = clw[0,0,1][0]
		with interval(1,-2):
			rew = reliq_def
			rei = reice_def
			rer = rrain_def
			res = rsnow_def
			tem2d = min(1.0, max(0.0,(con_ttp - tlyr[0,0,1])*0.05))
			if lcrick:
				clwf = 0.25 * clw[0,0,0][0] + 0.5 * clw[0,0,1][0] + 0.25 * clw[0,0,2][0]
			else:
				clwf = clw[0,0,1][0]
		with interval(-2,-1):
			rew = reliq_def
			rei = reice_def
			rer = rrain_def
			res = rsnow_def
			tem2d = min(1.0, max(0.0,(con_ttp - tlyr[0,0,1])*0.05))
			if lcrick:
				clwf = 0.75 * clw[0,0,1][0] + 0.25 * clw[0,0,0][0]
			else:
				clwf = clw[0,0,1][0]

	with computation(FORWARD), interval(0,1):
		rxlat = abs(xlat[0] / con_pi)
		ptop1[0,0][0] = ptopc_0_0 + (ptopc_0_1 - ptopc_0_0) * max(0.0, 4.0 * rxlat[0,0] - 1.0)
		ptop1[0,0][1] = ptopc_1_0 + (ptopc_1_1 - ptopc_1_0) * max(0.0, 4.0 * rxlat[0,0] - 1.0)
		ptop1[0,0][2] = ptopc_2_0 + (ptopc_2_1 - ptopc_2_0) * max(0.0, 4.0 * rxlat[0,0] - 1.0)
		ptop1[0,0][3] = ptopc_3_0 + (ptopc_3_1 - ptopc_3_0) * max(0.0, 4.0 * rxlat[0,0] - 1.0)
		if iovr == 3:
			de_lgth = max(0.6, 2.78 - 4.6 * rxlat[0,0])

	with computation(PARALLEL), interval(0,-1):
		clwt = max(0.0, (clwf[0,0,0] + cnvw[0,0,1])) * gfac * delp[0,0,1]
		cip = clwt * tem2d[0,0,0]
		cwp = clwt - cip[0,0,0]
		if slmsk[0] > 0.5 and slmsk[0] < 1.5:
			rew = 5.0 + 5.0 * tem2d[0,0,0]
		if cldtot[0,0,1] < climit:
			cwp = 0.0
			cip = 0.0
		if cldtot[0,0,1] >= climit and lcnorm:
			tem1 = 1.0 / max(climit2, cldtot[0,0,1])
			cwp = cwp * tem1
			cip = cip * tem1
		if cip[0,0,0] > 0.0:
			tem2 = tlyr[0,0,1] - con_ttp
			tem3 = (gord * cip[0,0,0] * plyr[0,0,1] / (delp[0,0,1] * tvly[0,0,1]))
			if tem2 < -50.0:
				rei = (1250.0 / 9.917) * tem3 ** 0.109
			elif tem2 < -40.0:
				rei = (1250.0 / 9.337) * tem3 ** 0.08
			elif tem2 < -30.0:
				rei = (1250.0 / 9.208) * tem3 ** 0.055
			else:
				rei = (1250.0 / 9.387) * tem3 ** 0.031
			rei = max(10.0, min(rei[0,0,0], 150.0))
	
	with computation(PARALLEL), interval(1,None):
		clouds[0,0,0][0] = cldtot[0,0,0]
		clouds[0,0,0][1] = cwp[0,0,-1]
		clouds[0,0,0][2] = rew[0,0,-1]
		clouds[0,0,0][3] = cip[0,0,-1]
		clouds[0,0,0][4] = rei[0,0,-1]
		clouds[0,0,0][6] = rer[0,0,-1]
		clouds[0,0,0][8] = res[0,0,-1]