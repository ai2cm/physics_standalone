import numpy as np
import xarray as xr
import os
import sys

from gt4py.gtscript import (
    stencil,
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
from phys_const import con_eps, con_epsm1, con_rocp, con_fvirt, con_rog, con_epsq
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
