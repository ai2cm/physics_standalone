import numpy as numpy
from config import *
from phys_const import rhowater


class GFS_control_type:
    def __init__(self):
        self.me = 0  # MPI rank designator
        self.master = 0  # MPI rank of master atmosphere processor
        self.nlunit = 0  # unit for namelist
        self.fn_nml = 0  # namelist filename for surface data cycling
        self.input_nml_file = " "  # character string containing full namelist
        self.fhzero = 0.0  # hours between clearing of diagnostic buckets
        self.ldiag3d = False  # flag for 3d diagnostic fields
        self.lssav = False  # logical flag for storing diagnostics
        self.fhcyc = 0.0  # frequency for surface data cycling (hours)
        self.thermodyn_id = 1  # valid for GFS only for get_prs/phi
        self.sfcpress_id = 1  # valid for GFS only for get_prs/phi
        self.gen_coord_hybrid = False  # for Henry's gen coord
        self.sfc_override = False  # use idealized surface conditions

        # --- set some grid extent parameters
        self.isc = 0  # starting i-index for this MPI-domain
        self.jsc = 0  # starting j-index for this MPI-domain
        self.nx = 0  # number of points in the i-dir for this MPI-domain
        self.ny = 0  # number of points in the j-dir for this MPI-domain
        self.levs = 0  # number of vertical levels
        self.cnx = 0  # number of points in the i-dir for this cubed-sphere face
        self.cny = 0  # number of points in the j-dir for this cubed-sphere face
        self.lonr = 0  # number of global points in x-dir (i) along the equator
        self.latr = 0  # number of global points in y-dir (j) along any meridian
        self.tile_num = 0
        self.nblks = 0  # for explicit data blocking: number of blocks
        self.blksz = 0  # for explicit data blocking: block sizes of all blocks

        # --- coupling parameters
        self.cplflx = False  # default no cplflx collection
        self.cplwav = False  # default no cplwav collection
        self.cplchm = False  # default no cplchm collection

        # --- integrated dynamics through earth's atmosphere
        self.lsidea = False

        # vay 2018  GW physics switches

        self.ldiag_ugwp = False
        self.do_ugwp = False  # do mesoscale UGWP + TOFD + RF
        self.do_tofd = False  # tofd flag in gwdps.f
        self.do_gwd = False  # logical for gravity wave drag (gwd)
        self.do_cnvgwd = False  # logical for convective gwd

        # --- calendars and time parameters and activation triggers
        self.dtp = 0.0  # physics timestep in seconds
        self.dtf = 0.0  # dynamics timestep in seconds
        self.nscyc = 0  # trigger for surface data cycling
        self.nszero = 0  # trigger for zeroing diagnostic buckets
        self.idat = np.zeros(8, dtype=DTYPE_INT)  # initialization date and time
        # (yr, mon, day, t-zone, hr, min, sec, mil-sec)
        self.idate = np.zeros(
            4, dtype=DTYPE_INT
        )  # initial date with different size and ordering
        # (hr, mon, day, yr)
        # --- radiation control parameters
        self.fhswr = 3600.0  # frequency for shortwave radiation (secs)
        self.fhlwr = 3600.0  # frequency for longwave radiation (secs)
        self.nsswr = 0  # integer trigger for shortwave radiation
        self.nslwr = 0  # integer trigger for longwave  radiation
        self.levr = -99  # number of vertical levels for radiation calculations
        self.nfxr = 39 + 6  # second dimension for fluxr diagnostic variable (radiation)
        self.aero_in = False  # flag for initializing aerosol data
        self.lmfshal = False  # parameter for radiation
        self.lmfdeep2 = False  # parameter for radiation
        self.nrcm = 0  # second dimension of random number stream for RAS
        self.iflip = 1  # iflip - is not the same as flipv
        self.isol = 0  # use prescribed solar constant
        self.ico2 = 0  # prescribed global mean value (old opernl)
        self.ialb = 0  # use climatology alb, based on sfc type
        # 1 => use modis based alb
        self.iems = 0  # use fixed value of 1.0
        self.iaer = 1  # default aerosol effect in sw only
        self.icliq_sw = 1  # sw optical property for liquid clouds
        self.iovr_sw = 1  # sw: max-random overlap clouds
        self.iovr_lw = 1  # lw: max-random overlap clouds
        self.ictm = 1  # ictm=0 => use data at initial cond time, if not
        #           available; use latest; no extrapolation.
        # ictm=1 => use data at the forecast time, if not
        #           available; use latest; do extrapolation.
        # ictm=yyyy0 => use yyyy data for the forecast time;
        #           no extrapolation.
        # ictm=yyyy1 = > use yyyy data for the fcst. If needed,
        #           do extrapolation to match the fcst time.
        # ictm=-1 => use user provided external data for
        #           the fcst time; no extrapolation.
        # ictm=-2 => same as ictm=0, but add seasonal cycle
        #           from climatology; no extrapolation.
        self.isubc_sw = 0  # sw clouds without sub-grid approximation
        self.isubc_lw = 0  # lw clouds without sub-grid approximation
        # =1 => sub-grid cloud with prescribed seeds
        # =2 => sub-grid cloud with randomly generated
        # seeds
        self.crick_proof = False  # CRICK-Proof cloud water
        self.ccnorm = False  # Cloud condensate normalized by cloud cover
        self.norad_precip = False  # radiation precip flag for Ferrier/Moorthi
        self.lwhtr = True  # flag to output lw heating rate (Radtend%lwhc)
        self.swhtr = True  # flag to output sw heating rate (Radtend%swhc)
        self.do_only_clearsky_rad = (
            False  # flag for whether to do only clear-sky radiation
        )

        # --- microphysical switch
        self.ncld = 1  # choice of cloud scheme
        # --- new microphysical switch
        self.imp_physics = 99  # choice of microphysics scheme
        self.imp_physics_gfdl = 11  # choice of GFDL     microphysics scheme
        self.imp_physics_thompson = 8  # choice of Thompson microphysics scheme
        self.imp_physics_wsm6 = 6  # choice of WSMG     microphysics scheme
        self.imp_physics_zhao_carr = 99  # choice of Zhao-Carr microphysics scheme
        self.imp_physics_zhao_carr_pdf = (
            98  # choice of Zhao-Carr microphysics scheme with PDF clouds
        )
        self.imp_physics_mg = 10  # choice of Morrison-Gettelman microphysics scheme
        # --- Z-C microphysical parameters
        self.psautco = [6.0e-4, 3.0e-4]  # [in] auto conversion coeff from ice to snow
        self.prautco = [1.0e-4, 1.0e-4]  # [in] auto conversion coeff from cloud to rain
        self.evpco = 2.0e-5  # [in] coeff for evaporation of largescale rain
        self.wminco = [1.0e-5, 1.0e-5]  # [in] water and ice minimum threshold for Zhao
        self.avg_max_length = 3600.0  # reset time in seconds for max hourly fields
        # --- M-G microphysical parameters
        self.fprcp = 0  # no prognostic rain and snow (MG)
        self.pdfflag = 4  # pdf flag for MG macrophysics
        self.mg_dcs = 200.0  # Morrison-Gettelman microphysics parameters
        self.mg_qcvar = 1.0
        self.mg_ts_auto_ice = [180.0, 180.0]  # ice auto conversion time scale

        self.mg_ncnst = 100.0e6  # constant droplet num concentration (m-3)
        self.mg_ninst = 0.15e6  # constant ice num concentration (m-3)
        self.mg_ngnst = 0.10e6  # constant graupel/hail num concentration (m-3)
        self.mg_berg_eff_factor = 0.0  # berg efficiency factor
        self.mg_alf = 1.0  # tuning factor for alphs in MG macrophysics
        self.mg_qcmin = [
            1.0e-9,
            1.0e-9,
        ]  # min liquid and ice mixing ratio in Mg macro clouds
        self.mg_precip_frac_method = (
            "max_overlap"  # type of precipitation fraction method
        )

        self.effr_in = False  # eg to turn on ffective radii for MG
        self.microp_uniform = True
        self.do_cldliq = True
        self.do_cldice = True
        self.hetfrz_classnuc = False

        self.mg_nccons = False
        self.mg_nicons = False
        self.mg_ngcons = False
        self.sed_supersat = True
        self.do_sb_physics = True
        self.mg_do_graupel = True
        self.mg_do_hail = False
        self.mg_do_ice_gmao = False
        self.mg_do_liq_liu = True

        self.shoc_parm = np.zeros(
            5
        )  # critical pressure in Pa for tke dissipation in shoc
        self.ncnd = 0  # number of cloud condensate types

        # --- Thompson's microphysical parameters
        self.ltaerosol = False  # flag for aerosol version
        self.lradar = False  # flag for radar reflectivity
        self.ttendlim = -999.0  # temperature tendency limiter per time step in K/s

        # --- GFDL microphysical paramters
        self.lgfdlmprad = False  # flag for GFDL mp scheme and radiation consistency
        self.do_gfdl_mp_in_physics = (
            True  # flag for whether to call gfdl_cloud_microphys_driver
        )

        # --- land/surface model parameters
        self.lsm = 1  # flag for land surface model lsm=1 for noah lsm
        self.lsm_noah = 1  # flag for NOAH land surface model
        self.lsm_noahmp = 2  # flag for NOAH land surface model
        self.lsm_ruc = 3  # flag for RUC land surface model
        self.lsoil = 4  # number of soil layers
        self.ivegsrc = 2  # ivegsrc = 0   => USGS,
        # ivegsrc = 1   => IGBP (20 category)
        # ivegsrc = 2   => UMD  (13 category)
        self.isot = 0  # isot = 0   => Zobler soil type  ( 9 category)
        # isot = 1   => STATSGO soil type (19 category)
        # -- the Noah MP options

        self.iopt_dveg = 4  # 1-> off table lai 2-> on 3-> off;4->off;5 -> on
        self.iopt_crs = 1  # canopy stomatal resistance (1-> ball-berry; 2->jarvis)
        self.iopt_btr = 1  # soil moisture factor for stomatal resistance (1-> noah; 2-> clm; 3-> ssib)
        self.iopt_run = (
            3  # runoff and groundwater (1->simgm; 2->simtop; 3->schaake96; 4->bats)
        )
        self.iopt_sfc = 1  # surface layer drag coeff (ch & cm) (1->m-o; 2->chen97)
        self.iopt_frz = 1  # supercooled liquid water (1-> ny06; 2->koren99)
        self.iopt_inf = 1  # frozen soil permeability (1-> ny06; 2->koren99)
        self.iopt_rad = (
            3  # radiation transfer (1->gap=f(3d,cosz); 2->gap=0; 3->gap=1-fveg)
        )
        self.iopt_alb = 2  # snow surface albedo (1->bats; 2->class)
        self.iopt_snf = 1  # rainfall & snowfall (1-jordan91; 2->bats; 3->noah)
        self.iopt_tbot = 2  # lower boundary of soil temperature (1->zero-flux; 2->noah)
        self.iopt_stc = 1  # snow/soil temperature time scheme (only layer 1)

        self.use_ufo = False  # flag for gcycle surface option
        self.use_analysis_sst = (
            False  # whether to set physics SST to dynamical core ts, which is
        )
        # equal to analysis SST when nudging is active
        # --- tuning parameters for physical parameterizations
        self.ras = False  # flag for ras convection scheme
        self.flipv = True  # flag for vertical direction flip (ras)
        # .true. implies surface at k=1
        self.trans_trac = (
            False  # flag for convective transport of tracers (RAS, CS, or SAMF)
        )
        self.old_monin = False  # flag for diff monin schemes
        self.cnvgwd = False  # flag for conv gravity wave drag

        self.mstrat = False  # flag for moorthi approach for stratus
        self.moist_adj = False  # flag for moist convective adjustment
        self.cscnv = False  # flag for Chikira-Sugiyama convection
        self.cal_pre = False  # flag controls precip type algorithm
        self.do_aw = False  # AW scale-aware option in cs convection
        self.do_awdd = False  # AW scale-aware option in cs convection
        self.flx_form = False  # AW scale-aware option in cs convection
        self.do_shoc = False  # flag for SHOC
        self.shocaftcnv = False  # flag for SHOC
        self.shoc_cld = False  # flag for clouds
        self.uni_cld = False  # flag for clouds in grrad
        self.h2o_phys = False  # flag for stratosphere h2o
        self.pdfcld = False  # flag for pdfcld
        self.shcnvcw = False  # flag for shallow convective cloud
        self.redrag = False  # flag for reduced drag coeff. over sea
        self.hybedmf = False  # flag for hybrid edmf pbl scheme
        self.satmedmf = False  # flag for scale-aware TKE-based moist edmf
        # vertical turbulent mixing scheme
        self.shinhong = (
            False  # flag for scale-aware Shinhong vertical turbulent mixing scheme
        )
        self.do_ysu = False  # flag for YSU turbulent mixing scheme
        self.dspheat = False  # flag for tke dissipative heating
        self.lheatstrg = False  # flag for canopy heat storage parameterization
        self.cnvcld = False
        self.random_clds = False  # flag controls whether clouds are random
        self.shal_cnv = False  # flag for calling shallow convection
        self.do_deep = False  # whether to do deep convection
        self.imfshalcnv = 1  # flag for mass-flux shallow convection scheme
        #     1: July 2010 version of mass-flux shallow conv scheme
        #         current operational version as of 2016
        #     2: scale- & aerosol-aware mass-flux shallow conv scheme (2017)
        #     3: scale- & aerosol-aware Grell-Freitas scheme (GSD)
        #     4: New Tiedtke scheme (CAPS)
        #     0: modified Tiedtke's eddy-diffusion shallow conv scheme
        #    -1: no shallow convection used
        self.imfdeepcnv = 1  # flag for mass-flux deep convection scheme
        #     1: July 2010 version of SAS conv scheme
        #           current operational version as of 2016
        #     2: scale- & aerosol-aware mass-flux deep conv scheme (2017)
        #     3: scale- & aerosol-aware Grell-Freitas scheme (GSD)
        #     4: New Tiedtke scheme (CAPS)
        #     0: old SAS Convection scheme before July 2010
        self.isatmedmf = 0  # flag for scale-aware TKE-based moist edmf scheme
        #     0: initial version of satmedmf (Nov. 2018)
        #     1: updated version of satmedmf (as of May 2019)

        self.nmtvr = 14  # number of topographic variables such as variance etc
        # used in the GWD parameterization
        self.jcap = 1  # number of spectral wave trancation used only by sascnv shalcnv
        self.cs_parm = [
            8.0,
            4.0,
            1.0e3,
            3.5e3,
            20.0,
            1.0,
            -999.0,
            1.0,
            0.6,
            0.0,
        ]  # tunable parameters for Chikira-Sugiyama convection
        self.flgmin = [0.180, 0.220]  # [in] ice fraction bounds
        self.cgwf = [0.5e0, 0.05e0]  # multiplication factor for convective GWD
        self.ccwf = [1.0e0, 1.0e0]  # multiplication factor for critical cloud
        # workfunction for RAS
        self.cdmbgwd = [
            2.0e0,
            0.25e0,
            1.0e0,
            1.0e0,
        ]  # multiplication factors for cdmb, gwd and NS gwd, tke based enhancement
        self.sup = 1.0  # supersaturation in pdf cloud when t is very low
        self.ctei_rm = [
            10.0e0,
            10.0e0,
        ]  # critical cloud top entrainment instability criteria
        # (used if mstrat=.true.)
        self.crtrh = [
            0.90e0,
            0.90e0,
            0.90e0,
        ]  # critical relative humidity at the surface
        # PBL top and at the top of the atmosphere
        self.dlqf = [0.0, 0.0]  # factor for cloud condensate detrainment
        # from cloud edges for RAS
        self.psauras = [
            1.0e-3,
            1.0e-3,
        ]  # [in] auto conversion coeff from ice to snow in ras
        self.prauras = [
            2.0e-3,
            2.0e-3,
        ]  # [in] auto conversion coeff from cloud to rain in ras
        self.wminras = [1.0e-5, 1.0e-5]  # [in] water and ice minimum threshold for ras

        self.seed0 = 0  # random seed for radiation

        self.rbcr = 0.25  # Critical Richardson Number in the PBL scheme

        # --- Rayleigh friction
        self.prslrd0 = 0.0  # pressure level from which Rayleigh Damping is applied
        self.ral_ts = 0.0  # time scale for Rayleigh damping in days

        # --- mass flux deep convection
        self.clam_deep = 0.1  # c_e for deep convection (Han and Pan, 2011, eq(6))
        self.c0s_deep = 0.002  # convective rain conversion parameter
        self.c1_deep = 0.002  # conversion parameter of detrainment from liquid water into grid-scale cloud water
        self.betal_deep = 0.05  # fraction factor of downdraft air mass reaching ground surface over land
        self.betas_deep = 0.05  # fraction factor of downdraft air mass reaching ground surface over sea
        self.evfact_deep = 0.3  # evaporation factor from convective rain
        self.evfactl_deep = 0.3  # evaporation factor from convective rain over land
        self.pgcon_deep = 0.55  # reduction factor in momentum transport due to convection induced pressure gradient force
        # 0.7 : Gregory et al. (1997, QJRMS)
        # 0.55: Zhang & Wu (2003, JAS)
        self.asolfac_deep = 0.958  # aerosol-aware parameter based on Lim (2011)
        # asolfac= cx / c0s(=.002)
        # cx = min([-0.7 ln(Nccn) + 24]*1.e-4, c0s)
        # Nccn: CCN number concentration in cm^(-3)
        # Until a realistic Nccn is provided, Nccns are assumed
        # as Nccn=100 for sea and Nccn=1000 for land

        # --- mass flux shallow convection
        self.clam_shal = 0.3  # c_e for shallow convection (Han and Pan, 2011, eq(6))
        self.c0s_shal = 0.002  # convective rain conversion parameter
        self.c1_shal = 5.0e-4  # conversion parameter of detrainment from liquid water into grid-scale cloud water
        self.pgcon_shal = 0.55  # reduction factor in momentum transport due to convection induced pressure gradient force
        # 0.7 : Gregory et al. (1997, QJRMS)
        # 0.55: Zhang & Wu (2003, JAS)
        self.asolfac_shal = 0.958  # aerosol-aware parameter based on Lim (2011)
        # asolfac= cx / c0s(=.002)
        # cx = min([-0.7 ln(Nccn) + 24]*1.e-4, c0s)
        # Nccn: CCN number concentration in cm^(-3)
        # Until a realistic Nccn is provided, Nccns are assumed
        # as Nccn=100 for sea and Nccn=1000 for land

        # --- near surface temperature model
        self.nst_anl = False  # flag for NSSTM analysis in gcycle/sfcsub
        self.lsea = 0
        self.nstf_name = [
            0,
            0,
            1,
            0,
            5,
        ]  # flag 0 for no nst  1 for uncoupled nst  and 2 for coupled NST
        # nstf_name contains the NSST related parameters
        # nstf_name(1) : 0 = NSSTM off, 1 = NSSTM on but uncoupled
        #                2 = NSSTM on and coupled
        # nstf_name(2) : 1 = NSSTM spin up on, 0 = NSSTM spin up off
        # nstf_name(3) : 1 = NSST analysis on, 0 = NSSTM analysis off
        # nstf_name(4) : zsea1 in mm
        # nstf_name(5) : zsea2 in mm
        # --- fractional grid
        self.frac_grid = False  # flag for fractional grid
        self.min_lakeice = 0.15  # minimum lake ice value
        self.min_seaice = 1.0e-6  # minimum sea  ice value
        self.rho_h2o = rhowater  # density of fresh water

        # --- surface layer z0 scheme
        self.sfc_z0_type = 0  # surface roughness options over ocean:
        # 0=no change
        # 6=areodynamical roughness over water with input 10-m wind
        # 7=slightly decrease Cd for higher wind speed compare to 6

        # --- background vertical diffusion
        self.xkzm_m = (
            1.0  # [in] bkgd_vdif_m  background vertical diffusion for momentum
        )
        self.xkzm_h = 1.0  # [in] bkgd_vdif_h  background vertical diffusion for heat q
        self.xkzm_s = (
            1.0  # [in] bkgd_vdif_s  sigma threshold for background mom. diffusion
        )
        self.xkzminv = 0.3  # diffusivity in inversion layers
        self.moninq_fac = 1.0  # turbulence diffusion coefficient factor
        self.dspfac = 1.0  # tke dissipative heating factor
        self.bl_upfr = 0.13  # updraft fraction in boundary layer mass flux scheme
        self.bl_dnfr = 0.1  # downdraft fraction in boundary layer mass flux scheme

        # ---cellular automata control parameters
        self.nca = 1  # number of independent cellular automata
        self.nlives = 10  # cellular automata lifetime
        self.ncells = 5  # cellular automata finer grid
        self.nfracseed = 0.5  # cellular automata seed probability
        self.nseed = 100000  # cellular automata seed frequency
        self.do_ca = False  # cellular automata main switch
        self.ca_sgs = False  # switch for sgs ca
        self.ca_global = False  # switch for global ca
        self.ca_smooth = False  # switch for gaussian spatial filter
        self.isppt_deep = False  # switch for combination with isppt_deep. OBS! Switches off SPPT on other tendencies!
        self.iseed_ca = 0  # seed for random number generation in ca scheme
        self.nspinup = 1  # number of iterations to spin up the ca
        self.nthresh = 0.0  # threshold used for perturbed vertical velocity

        # --- stochastic physics control parameters
        self.do_sppt = False
        self.use_zmtnblck = False
        self.do_shum = False
        self.do_skeb = False
        self.skeb_npass = 11
        self.do_sfcperts = False
        self.nsfcpert = 6
        self.pertz0 = -999  # mg, sfc-perts
        self.pertzt = -999  # mg, sfc-perts
        self.pertshc = -999  # mg, sfc-perts
        self.pertlai = -999  # mg, sfc-perts
        self.pertalb = -999  # mg, sfc-perts
        self.pertvegf = -999  # mg, sfc-perts
        # --- tracer handling
        self.tracer_names = ""  # array of initialized tracers from dynamic core
        self.ntrac = 0  # number of tracers

        self.ntoz = 0  # tracer index for ozone mixing ratio
        self.ntcw = 0  # tracer index for cloud condensate (or liquid water)
        self.ntiw = 0  # tracer index for ice water
        self.ntrw = 0  # tracer index for rain water
        self.ntsw = 0  # tracer index for snow water
        self.ntgl = 0  # tracer index for graupel
        self.ntclamt = 0  # tracer index for cloud amount
        self.ntlnc = 0  # tracer index for liquid number concentration
        self.ntinc = 0  # tracer index for ice    number concentration
        self.ntrnc = 0  # tracer index for rain   number concentration
        self.ntsnc = 0  # tracer index for snow   number concentration
        self.ntgnc = 0  # tracer index for graupel number concentration
        self.ntke = 0  # tracer index for kinetic energy
        self.nto = 0  # tracer index for oxygen ion
        self.nto2 = 0  # tracer index for oxygen
        self.ntwa = 0  # tracer index for water friendly aerosol
        self.ntia = 0  # tracer index for ice friendly aerosol
        self.ntchm = 0  # number of chemical tracers
        self.ntchs = 0  # tracer index for first chemical tracer
        self.ntdiag = False  # array to control diagnostics for chemical tracers
        self.fscav = 0.0  # array of aerosol scavenging coefficients

        # --- derived totals for phy_f*d
        self.ntot2d = 0  # total number of variables for phyf2d
        self.ntot3d = 0  # total number of variables for phyf3d
        self.indcld = (
            0  # location of cloud fraction in phyf3d (used only for SHOC or MG)
        )
        self.num_p2d = 0  # number of 2D arrays needed for microphysics
        self.num_p3d = 0  # number of 3D arrays needed for microphysics
        self.nshoc_2d = 0  # number of 2d fields for SHOC
        self.nshoc_3d = 0  # number of 3d fields for SHOC
        self.ncnvcld3d = 0  # number of convective 3d clouds fields
        self.npdf3d = (
            0  # number of 3d arrays associated with pdf based clouds/microphysics
        )
        self.nctp = 0  # number of cloud types in Chikira-Sugiyama scheme
        self.ncnvw = 0  # the index of cnvw in phy_f3d
        self.ncnvc = 0  # the index of cnvc in phy_f3d
        self.nleffr = 0  # the index of cloud liquid water effective radius in phy_f3d
        self.nieffr = 0  # the index of ice effective radius in phy_f3d
        self.nreffr = 0  # the index of rain effective radius in phy_f3d
        self.nseffr = 0  # the index of snow effective radius in phy_f3d
        self.ngeffr = 0  # the index of graupel effective radius in phy_f3d

        # --- debug flag
        self.debug = False
        self.pre_rad = False  # flag for testing purpose
        self.do_ocean = False  # flag for slab ocean model

        # --- variables modified at each time step
        self.ipt = 0  # index for diagnostic printout point
        self.lprnt = False  # control flag for diagnostic print out
        self.lsswr = False  # logical flags for sw radiation calls
        self.lslwr = False  # logical flags for lw radiation calls
        self.solhr = 0.0  # hour time after 00z at the t-step
        self.solcon = (
            0.0  # solar constant (sun-earth distant adjusted)  [set via radupdate]
        )
        self.slag = (
            0.0  # equation of time ( radian )                  [set via radupdate]
        )
        self.sdec = (
            0.0  # sin of the solar declination angle           [set via radupdate]
        )
        self.cdec = (
            0.0  # cos of the solar declination angle           [set via radupdate]
        )
        self.clstp = 0.0  # index used by cnvc90 (for convective clouds)
        # legacy stuff - does not affect forecast
        self.phour = 0.0  # previous forecast hour
        self.fhour = 0.0  # current forecast hour
        self.zhour = 0.0  # previous hour diagnostic buckets emptied
        self.kdt = 0  # current forecast iteration

        self.jdat = np.zeros(8)  # current forecast date and time
        # (yr, mon, day, t-zone, hr, min, sec, mil-sec)
        self.imn = 0  # current forecast month
        self.julian = 0.0  # current forecast julian date
        self.yearlen = 0  # current length of the year

        self.iccn = False  # using IN CCN forcing for MG2/3

        # --- IAU
        self.iau_offset = 0
        self.iau_delthrs = 0.0  # iau time interval (to scale increments) in hours
        self.iau_inc_files = np.zeros(7)  # list of increment files
        self.iaufhrs = -1 * np.ones(7)  # forecast hours associated with increment files
        self.iau_filter_increments = False
        self.sst_perturbation = 0.0  # Sea surface temperature perturbation to climatology or nudging SST (default 0.0 K)
