import numpy as numpy
import yaml
from config import *
from phys_const import rhowater


class GFS_control_type:
    def __init__(self):

        data = {}

        data["me"] = 0  # MPI rank designator
        data["master"] = 0  # MPI rank of master atmosphere processor
        data["nlunit"] = 0  # unit for namelist
        data["fn_nml"] = 0  # namelist filename for surface data cycling
        data["input_nml_file"] = " "  # character string containing full namelist
        data["fhzero"] = 0.0  # hours between clearing of diagnostic buckets
        data["ldiag3d"] = False  # flag for 3d diagnostic fields
        data["lssav"] = False  # logical flag for storing diagnostics
        data["fhcyc"] = 0.0  # frequency for surface data cycling (hours)
        data["thermodyn_id"] = 1  # valid for GFS only for get_prs/phi
        data["sfcpress_id"] = 1  # valid for GFS only for get_prs/phi
        data["gen_coord_hybrid"] = False  # for Henry's gen coord
        data["sfc_override"] = False  # use idealized surface conditions

        # --- set some grid extent parameters
        data["isc"] = 0  # starting i-index for this MPI-domain
        data["jsc"] = 0  # starting j-index for this MPI-domain
        data["nx"] = 0  # number of points in the i-dir for this MPI-domain
        data["ny"] = 0  # number of points in the j-dir for this MPI-domain
        data["levs"] = 0  # number of vertical levels
        data["cnx"] = 0  # number of points in the i-dir for this cubed-sphere face
        data["cny"] = 0  # number of points in the j-dir for this cubed-sphere face
        data["lonr"] = 0  # number of global points in x-dir (i) along the equator
        data["latr"] = 0  # number of global points in y-dir (j) along any meridian
        data["tile_num"] = 0
        data["nblks"] = 0  # for explicit data blocking: number of blocks
        data["blksz"] = 0  # for explicit data blocking: block sizes of all blocks

        # --- coupling parameters
        data["cplflx"] = False  # default no cplflx collection
        data["cplwav"] = False  # default no cplwav collection
        data["cplchm"] = False  # default no cplchm collection

        # --- integrated dynamics through earth's atmosphere
        data["lsidea"] = False

        # vay 2018  GW physics switches

        data["ldiag_ugwp"] = False
        data["do_ugwp"] = False  # do mesoscale UGWP + TOFD + RF
        data["do_tofd"] = False  # tofd flag in gwdps.f
        data["do_gwd"] = False  # logical for gravity wave drag (gwd)
        data["do_cnvgwd"] = False  # logical for convective gwd

        # --- calendars and time parameters and activation triggers
        data["dtp"] = 0.0  # physics timestep in seconds
        data["dtf"] = 0.0  # dynamics timestep in seconds
        data["nscyc"] = 0  # trigger for surface data cycling
        data["nszero"] = 0  # trigger for zeroing diagnostic buckets
        data["idat"] = np.zeros(8, dtype=DTYPE_INT)  # initialization date and time
        # (yr, mon, day, t-zone, hr, min, sec, mil-sec)
        data["idate"] = np.zeros(
            4, dtype=DTYPE_INT
        )  # initial date with different size and ordering
        # (hr, mon, day, yr)
        # --- radiation control parameters
        data["fhswr"] = 3600.0  # frequency for shortwave radiation (secs)
        data["fhlwr"] = 3600.0  # frequency for longwave radiation (secs)
        data["nsswr"] = 0  # integer trigger for shortwave radiation
        data["nslwr"] = 0  # integer trigger for longwave  radiation
        data["levr"] = -99  # number of vertical levels for radiation calculations
        data["nfxr"] = (
            39 + 6
        )  # second dimension for fluxr diagnostic variable (radiation)
        data["aero_in"] = False  # flag for initializing aerosol data
        data["lmfshal"] = False  # parameter for radiation
        data["lmfdeep2"] = False  # parameter for radiation
        data["nrcm"] = 0  # second dimension of random number stream for RAS
        data["iflip"] = 1  # iflip - is not the same as flipv
        data["isol"] = 0  # use prescribed solar constant
        data["ico2"] = 0  # prescribed global mean value (old opernl)
        data["ialb"] = 0  # use climatology alb, based on sfc type
        # 1 => use modis based alb
        data["iems"] = 0  # use fixed value of 1.0
        data["iaer"] = 1  # default aerosol effect in sw only
        data["icliq_sw"] = 1  # sw optical property for liquid clouds
        data["iovr_sw"] = 1  # sw: max-random overlap clouds
        data["iovr_lw"] = 1  # lw: max-random overlap clouds
        data["ictm"] = 1  # ictm=0 => use data at initial cond time, if not
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
        data["isubc_sw"] = 0  # sw clouds without sub-grid approximation
        data["isubc_lw"] = 0  # lw clouds without sub-grid approximation
        # =1 => sub-grid cloud with prescribed seeds
        # =2 => sub-grid cloud with randomly generated
        # seeds
        data["crick_proof"] = False  # CRICK-Proof cloud water
        data["ccnorm"] = False  # Cloud condensate normalized by cloud cover
        data["norad_precip"] = False  # radiation precip flag for Ferrier/Moorthi
        data["lwhtr"] = True  # flag to output lw heating rate (Radtend%lwhc)
        data["swhtr"] = True  # flag to output sw heating rate (Radtend%swhc)
        data[
            "do_only_clearsky_rad"
        ] = False  # flag for whether to do only clear-sky radiation

        # --- microphysical switch
        data["ncld"] = 1  # choice of cloud scheme
        # --- new microphysical switch
        data["imp_physics"] = 99  # choice of microphysics scheme
        data["imp_physics_gfdl"] = 11  # choice of GFDL     microphysics scheme
        data["imp_physics_thompson"] = 8  # choice of Thompson microphysics scheme
        data["imp_physics_wsm6"] = 6  # choice of WSMG     microphysics scheme
        data["imp_physics_zhao_carr"] = 99  # choice of Zhao-Carr microphysics scheme
        data[
            "imp_physics_zhao_carr_pdf"
        ] = 98  # choice of Zhao-Carr microphysics scheme with PDF clouds
        data["imp_physics_mg"] = 10  # choice of Morrison-Gettelman microphysics scheme
        # --- Z-C microphysical parameters
        data["psautco"] = [
            6.0e-4,
            3.0e-4,
        ]  # [in] auto conversion coeff from ice to snow
        data["prautco"] = [
            1.0e-4,
            1.0e-4,
        ]  # [in] auto conversion coeff from cloud to rain
        data["evpco"] = 2.0e-5  # [in] coeff for evaporation of largescale rain
        data["wminco"] = [
            1.0e-5,
            1.0e-5,
        ]  # [in] water and ice minimum threshold for Zhao
        data["avg_max_length"] = 3600.0  # reset time in seconds for max hourly fields
        # --- M-G microphysical parameters
        data["fprcp"] = 0  # no prognostic rain and snow (MG)
        data["pdfflag"] = 4  # pdf flag for MG macrophysics
        data["mg_dcs"] = 200.0  # Morrison-Gettelman microphysics parameters
        data["mg_qcvar"] = 1.0
        data["mg_ts_auto_ice"] = [180.0, 180.0]  # ice auto conversion time scale

        data["mg_ncnst"] = 100.0e6  # constant droplet num concentration (m-3)
        data["mg_ninst"] = 0.15e6  # constant ice num concentration (m-3)
        data["mg_ngnst"] = 0.10e6  # constant graupel/hail num concentration (m-3)
        data["mg_berg_eff_factor"] = 0.0  # berg efficiency factor
        data["mg_alf"] = 1.0  # tuning factor for alphs in MG macrophysics
        data["mg_qcmin"] = [
            1.0e-9,
            1.0e-9,
        ]  # min liquid and ice mixing ratio in Mg macro clouds
        data[
            "mg_precip_frac_method"
        ] = "max_overlap"  # type of precipitation fraction method

        data["effr_in"] = False  # eg to turn on ffective radii for MG
        data["microp_uniform"] = True
        data["do_cldliq"] = True
        data["do_cldice"] = True
        data["hetfrz_classnuc"] = False

        data["mg_nccons"] = False
        data["mg_nicons"] = False
        data["mg_ngcons"] = False
        data["sed_supersat"] = True
        data["do_sb_physics"] = True
        data["mg_do_graupel"] = True
        data["mg_do_hail"] = False
        data["mg_do_ice_gmao"] = False
        data["mg_do_liq_liu"] = True

        data["shoc_parm"] = np.zeros(
            5
        )  # critical pressure in Pa for tke dissipation in shoc
        data["ncnd"] = 0  # number of cloud condensate types

        # --- Thompson's microphysical parameters
        data["ltaerosol"] = False  # flag for aerosol version
        data["lradar"] = False  # flag for radar reflectivity
        data["ttendlim"] = -999.0  # temperature tendency limiter per time step in K/s

        # --- GFDL microphysical paramters
        data["lgfdlmprad"] = False  # flag for GFDL mp scheme and radiation consistency
        data[
            "do_gfdl_mp_in_physics"
        ] = True  # flag for whether to call gfdl_cloud_microphys_driver

        # --- land/surface model parameters
        data["lsm"] = 1  # flag for land surface model lsm=1 for noah lsm
        data["lsm_noah"] = 1  # flag for NOAH land surface model
        data["lsm_noahmp"] = 2  # flag for NOAH land surface model
        data["lsm_ruc"] = 3  # flag for RUC land surface model
        data["lsoil"] = 4  # number of soil layers
        data["ivegsrc"] = 2  # ivegsrc = 0   => USGS,
        # ivegsrc = 1   => IGBP (20 category)
        # ivegsrc = 2   => UMD  (13 category)
        data["isot"] = 0  # isot = 0   => Zobler soil type  ( 9 category)
        # isot = 1   => STATSGO soil type (19 category)
        # -- the Noah MP options

        data["iopt_dveg"] = 4  # 1-> off table lai 2-> on 3-> off;4->off;5 -> on
        data["iopt_crs"] = 1  # canopy stomatal resistance (1-> ball-berry; 2->jarvis)
        data[
            "iopt_btr"
        ] = 1  # soil moisture factor for stomatal resistance (1-> noah; 2-> clm; 3-> ssib)
        data[
            "iopt_run"
        ] = 3  # runoff and groundwater (1->simgm; 2->simtop; 3->schaake96; 4->bats)
        data["iopt_sfc"] = 1  # surface layer drag coeff (ch & cm) (1->m-o; 2->chen97)
        data["iopt_frz"] = 1  # supercooled liquid water (1-> ny06; 2->koren99)
        data["iopt_inf"] = 1  # frozen soil permeability (1-> ny06; 2->koren99)
        data[
            "iopt_rad"
        ] = 3  # radiation transfer (1->gap=f(3d,cosz); 2->gap=0; 3->gap=1-fveg)
        data["iopt_alb"] = 2  # snow surface albedo (1->bats; 2->class)
        data["iopt_snf"] = 1  # rainfall & snowfall (1-jordan91; 2->bats; 3->noah)
        data[
            "iopt_tbot"
        ] = 2  # lower boundary of soil temperature (1->zero-flux; 2->noah)
        data["iopt_stc"] = 1  # snow/soil temperature time scheme (only layer 1)

        data["use_ufo"] = False  # flag for gcycle surface option
        data[
            "use_analysis_sst"
        ] = False  # whether to set physics SST to dynamical core ts, which is
        # equal to analysis SST when nudging is active
        # --- tuning parameters for physical parameterizations
        data["ras"] = False  # flag for ras convection scheme
        data["flipv"] = True  # flag for vertical direction flip (ras)
        # .true. implies surface at k=1
        data[
            "trans_trac"
        ] = False  # flag for convective transport of tracers (RAS, CS, or SAMF)
        data["old_monin"] = False  # flag for diff monin schemes
        data["cnvgwd"] = False  # flag for conv gravity wave drag

        data["mstrat"] = False  # flag for moorthi approach for stratus
        data["moist_adj"] = False  # flag for moist convective adjustment
        data["cscnv"] = False  # flag for Chikira-Sugiyama convection
        data["cal_pre"] = False  # flag controls precip type algorithm
        data["do_aw"] = False  # AW scale-aware option in cs convection
        data["do_awdd"] = False  # AW scale-aware option in cs convection
        data["flx_form"] = False  # AW scale-aware option in cs convection
        data["do_shoc"] = False  # flag for SHOC
        data["shocaftcnv"] = False  # flag for SHOC
        data["shoc_cld"] = False  # flag for clouds
        data["uni_cld"] = False  # flag for clouds in grrad
        data["h2o_phys"] = False  # flag for stratosphere h2o
        data["pdfcld"] = False  # flag for pdfcld
        data["shcnvcw"] = False  # flag for shallow convective cloud
        data["redrag"] = False  # flag for reduced drag coeff. over sea
        data["hybedmf"] = False  # flag for hybrid edmf pbl scheme
        data["satmedmf"] = False  # flag for scale-aware TKE-based moist edmf
        # vertical turbulent mixing scheme
        data[
            "shinhong"
        ] = False  # flag for scale-aware Shinhong vertical turbulent mixing scheme
        data["do_ysu"] = False  # flag for YSU turbulent mixing scheme
        data["dspheat"] = False  # flag for tke dissipative heating
        data["lheatstrg"] = False  # flag for canopy heat storage parameterization
        data["cnvcld"] = False
        data["random_clds"] = False  # flag controls whether clouds are random
        data["shal_cnv"] = False  # flag for calling shallow convection
        data["do_deep"] = False  # whether to do deep convection
        data["imfshalcnv"] = 1  # flag for mass-flux shallow convection scheme
        #     1: July 2010 version of mass-flux shallow conv scheme
        #         current operational version as of 2016
        #     2: scale- & aerosol-aware mass-flux shallow conv scheme (2017)
        #     3: scale- & aerosol-aware Grell-Freitas scheme (GSD)
        #     4: New Tiedtke scheme (CAPS)
        #     0: modified Tiedtke's eddy-diffusion shallow conv scheme
        #    -1: no shallow convection used
        data["imfdeepcnv"] = 1  # flag for mass-flux deep convection scheme
        #     1: July 2010 version of SAS conv scheme
        #           current operational version as of 2016
        #     2: scale- & aerosol-aware mass-flux deep conv scheme (2017)
        #     3: scale- & aerosol-aware Grell-Freitas scheme (GSD)
        #     4: New Tiedtke scheme (CAPS)
        #     0: old SAS Convection scheme before July 2010
        data["isatmedmf"] = 0  # flag for scale-aware TKE-based moist edmf scheme
        #     0: initial version of satmedmf (Nov. 2018)
        #     1: updated version of satmedmf (as of May 2019)

        data["nmtvr"] = 14  # number of topographic variables such as variance etc
        # used in the GWD parameterization
        data[
            "jcap"
        ] = 1  # number of spectral wave trancation used only by sascnv shalcnv
        data["cs_parm"] = [
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
        data["flgmin"] = [0.180, 0.220]  # [in] ice fraction bounds
        data["cgwf"] = [0.5e0, 0.05e0]  # multiplication factor for convective GWD
        data["ccwf"] = [1.0e0, 1.0e0]  # multiplication factor for critical cloud
        # workfunction for RAS
        data["cdmbgwd"] = [
            2.0e0,
            0.25e0,
            1.0e0,
            1.0e0,
        ]  # multiplication factors for cdmb, gwd and NS gwd, tke based enhancement
        data["sup"] = 1.0  # supersaturation in pdf cloud when t is very low
        data["ctei_rm"] = [
            10.0e0,
            10.0e0,
        ]  # critical cloud top entrainment instability criteria
        # (used if mstrat=.true.)
        data["crtrh"] = [
            0.90e0,
            0.90e0,
            0.90e0,
        ]  # critical relative humidity at the surface
        # PBL top and at the top of the atmosphere
        data["dlqf"] = [0.0, 0.0]  # factor for cloud condensate detrainment
        # from cloud edges for RAS
        data["psauras"] = [
            1.0e-3,
            1.0e-3,
        ]  # [in] auto conversion coeff from ice to snow in ras
        data["prauras"] = [
            2.0e-3,
            2.0e-3,
        ]  # [in] auto conversion coeff from cloud to rain in ras
        data["wminras"] = [
            1.0e-5,
            1.0e-5,
        ]  # [in] water and ice minimum threshold for ras

        data["seed0"] = 0  # random seed for radiation

        data["rbcr"] = 0.25  # Critical Richardson Number in the PBL scheme

        # --- Rayleigh friction
        data["prslrd0"] = 0.0  # pressure level from which Rayleigh Damping is applied
        data["ral_ts"] = 0.0  # time scale for Rayleigh damping in days

        # --- mass flux deep convection
        data["clam_deep"] = 0.1  # c_e for deep convection (Han and Pan, 2011, eq(6))
        data["c0s_deep"] = 0.002  # convective rain conversion parameter
        data[
            "c1_deep"
        ] = 0.002  # conversion parameter of detrainment from liquid water into grid-scale cloud water
        data[
            "betal_deep"
        ] = 0.05  # fraction factor of downdraft air mass reaching ground surface over land
        data[
            "betas_deep"
        ] = 0.05  # fraction factor of downdraft air mass reaching ground surface over sea
        data["evfact_deep"] = 0.3  # evaporation factor from convective rain
        data["evfactl_deep"] = 0.3  # evaporation factor from convective rain over land
        data[
            "pgcon_deep"
        ] = 0.55  # reduction factor in momentum transport due to convection induced pressure gradient force
        # 0.7 : Gregory et al. (1997, QJRMS)
        # 0.55: Zhang & Wu (2003, JAS)
        data["asolfac_deep"] = 0.958  # aerosol-aware parameter based on Lim (2011)
        # asolfac= cx / c0s(=.002)
        # cx = min([-0.7 ln(Nccn) + 24]*1.e-4, c0s)
        # Nccn: CCN number concentration in cm^(-3)
        # Until a realistic Nccn is provided, Nccns are assumed
        # as Nccn=100 for sea and Nccn=1000 for land

        # --- mass flux shallow convection
        data["clam_shal"] = 0.3  # c_e for shallow convection (Han and Pan, 2011, eq(6))
        data["c0s_shal"] = 0.002  # convective rain conversion parameter
        data[
            "c1_shal"
        ] = 5.0e-4  # conversion parameter of detrainment from liquid water into grid-scale cloud water
        data[
            "pgcon_shal"
        ] = 0.55  # reduction factor in momentum transport due to convection induced pressure gradient force
        # 0.7 : Gregory et al. (1997, QJRMS)
        # 0.55: Zhang & Wu (2003, JAS)
        data["asolfac_shal"] = 0.958  # aerosol-aware parameter based on Lim (2011)
        # asolfac= cx / c0s(=.002)
        # cx = min([-0.7 ln(Nccn) + 24]*1.e-4, c0s)
        # Nccn: CCN number concentration in cm^(-3)
        # Until a realistic Nccn is provided, Nccns are assumed
        # as Nccn=100 for sea and Nccn=1000 for land

        # --- near surface temperature model
        data["nst_anl"] = False  # flag for NSSTM analysis in gcycle/sfcsub
        data["lsea"] = 0
        data["nstf_name"] = [
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
        data["frac_grid"] = False  # flag for fractional grid
        data["min_lakeice"] = 0.15  # minimum lake ice value
        data["min_seaice"] = 1.0e-6  # minimum sea  ice value
        data["rho_h2o"] = rhowater  # density of fresh water

        # --- surface layer z0 scheme
        data["sfc_z0_type"] = 0  # surface roughness options over ocean:
        # 0=no change
        # 6=areodynamical roughness over water with input 10-m wind
        # 7=slightly decrease Cd for higher wind speed compare to 6

        # --- background vertical diffusion
        data[
            "xkzm_m"
        ] = 1.0  # [in] bkgd_vdif_m  background vertical diffusion for momentum
        data[
            "xkzm_h"
        ] = 1.0  # [in] bkgd_vdif_h  background vertical diffusion for heat q
        data[
            "xkzm_s"
        ] = 1.0  # [in] bkgd_vdif_s  sigma threshold for background mom. diffusion
        data["xkzminv"] = 0.3  # diffusivity in inversion layers
        data["moninq_fac"] = 1.0  # turbulence diffusion coefficient factor
        data["dspfac"] = 1.0  # tke dissipative heating factor
        data["bl_upfr"] = 0.13  # updraft fraction in boundary layer mass flux scheme
        data["bl_dnfr"] = 0.1  # downdraft fraction in boundary layer mass flux scheme

        # ---cellular automata control parameters
        data["nca"] = 1  # number of independent cellular automata
        data["nlives"] = 10  # cellular automata lifetime
        data["ncells"] = 5  # cellular automata finer grid
        data["nfracseed"] = 0.5  # cellular automata seed probability
        data["nseed"] = 100000  # cellular automata seed frequency
        data["do_ca"] = False  # cellular automata main switch
        data["ca_sgs"] = False  # switch for sgs ca
        data["ca_global"] = False  # switch for global ca
        data["ca_smooth"] = False  # switch for gaussian spatial filter
        data[
            "isppt_deep"
        ] = False  # switch for combination with isppt_deep. OBS! Switches off SPPT on other tendencies!
        data["iseed_ca"] = 0  # seed for random number generation in ca scheme
        data["nspinup"] = 1  # number of iterations to spin up the ca
        data["nthresh"] = 0.0  # threshold used for perturbed vertical velocity

        # --- stochastic physics control parameters
        data["do_sppt"] = False
        data["use_zmtnblck"] = False
        data["do_shum"] = False
        data["do_skeb"] = False
        data["skeb_npass"] = 11
        data["do_sfcperts"] = False
        data["nsfcpert"] = 6
        data["pertz0"] = -999  # mg, sfc-perts
        data["pertzt"] = -999  # mg, sfc-perts
        data["pertshc"] = -999  # mg, sfc-perts
        data["pertlai"] = -999  # mg, sfc-perts
        data["pertalb"] = -999  # mg, sfc-perts
        data["pertvegf"] = -999  # mg, sfc-perts
        # --- tracer handling
        data["tracer_names"] = ""  # array of initialized tracers from dynamic core
        data["ntrac"] = 0  # number of tracers

        data["ntoz"] = 0  # tracer index for ozone mixing ratio
        data["ntcw"] = 0  # tracer index for cloud condensate (or liquid water)
        data["ntiw"] = 0  # tracer index for ice water
        data["ntrw"] = 0  # tracer index for rain water
        data["ntsw"] = 0  # tracer index for snow water
        data["ntgl"] = 0  # tracer index for graupel
        data["ntclamt"] = 0  # tracer index for cloud amount
        data["ntlnc"] = 0  # tracer index for liquid number concentration
        data["ntinc"] = 0  # tracer index for ice    number concentration
        data["ntrnc"] = 0  # tracer index for rain   number concentration
        data["ntsnc"] = 0  # tracer index for snow   number concentration
        data["ntgnc"] = 0  # tracer index for graupel number concentration
        data["ntke"] = 0  # tracer index for kinetic energy
        data["nto"] = 0  # tracer index for oxygen ion
        data["nto2"] = 0  # tracer index for oxygen
        data["ntwa"] = 0  # tracer index for water friendly aerosol
        data["ntia"] = 0  # tracer index for ice friendly aerosol
        data["ntchm"] = 0  # number of chemical tracers
        data["ntchs"] = 0  # tracer index for first chemical tracer
        data["ntdiag"] = False  # array to control diagnostics for chemical tracers
        data["fscav"] = 0.0  # array of aerosol scavenging coefficients

        # --- derived totals for phy_f*d
        data["ntot2d"] = 0  # total number of variables for phyf2d
        data["ntot3d"] = 0  # total number of variables for phyf3d
        data[
            "indcld"
        ] = 0  # location of cloud fraction in phyf3d (used only for SHOC or MG)
        data["num_p2d"] = 0  # number of 2D arrays needed for microphysics
        data["num_p3d"] = 0  # number of 3D arrays needed for microphysics
        data["nshoc_2d"] = 0  # number of 2d fields for SHOC
        data["nshoc_3d"] = 0  # number of 3d fields for SHOC
        data["ncnvcld3d"] = 0  # number of convective 3d clouds fields
        data[
            "npdf3d"
        ] = 0  # number of 3d arrays associated with pdf based clouds/microphysics
        data["nctp"] = 0  # number of cloud types in Chikira-Sugiyama scheme
        data["ncnvw"] = 0  # the index of cnvw in phy_f3d
        data["ncnvc"] = 0  # the index of cnvc in phy_f3d
        data[
            "nleffr"
        ] = 0  # the index of cloud liquid water effective radius in phy_f3d
        data["nieffr"] = 0  # the index of ice effective radius in phy_f3d
        data["nreffr"] = 0  # the index of rain effective radius in phy_f3d
        data["nseffr"] = 0  # the index of snow effective radius in phy_f3d
        data["ngeffr"] = 0  # the index of graupel effective radius in phy_f3d

        # --- debug flag
        data["debug"] = False
        data["pre_rad"] = False  # flag for testing purpose
        data["do_ocean"] = False  # flag for slab ocean model

        # --- variables modified at each time step
        data["ipt"] = 0  # index for diagnostic printout point
        data["lprnt"] = False  # control flag for diagnostic print out
        data["lsswr"] = False  # logical flags for sw radiation calls
        data["lslwr"] = False  # logical flags for lw radiation calls
        data["solhr"] = 0.0  # hour time after 00z at the t-step
        data[
            "solcon"
        ] = 0.0  # solar constant (sun-earth distant adjusted)  [set via radupdate]
        data[
            "slag"
        ] = 0.0  # equation of time ( radian )                  [set via radupdate]
        data[
            "sdec"
        ] = 0.0  # sin of the solar declination angle           [set via radupdate]
        data[
            "cdec"
        ] = 0.0  # cos of the solar declination angle           [set via radupdate]
        data["clstp"] = 0.0  # index used by cnvc90 (for convective clouds)
        # legacy stuff - does not affect forecast
        data["phour"] = 0.0  # previous forecast hour
        data["fhour"] = 0.0  # current forecast hour
        data["zhour"] = 0.0  # previous hour diagnostic buckets emptied
        data["kdt"] = 0  # current forecast iteration

        data["jdat"] = np.zeros(8)  # current forecast date and time
        # (yr, mon, day, t-zone, hr, min, sec, mil-sec)
        data["imn"] = 0  # current forecast month
        data["julian"] = 0.0  # current forecast julian date
        data["yearlen"] = 0  # current length of the year

        data["iccn"] = False  # using IN CCN forcing for MG2/3

        # --- IAU
        data["iau_offset"] = 0
        data["iau_delthrs"] = 0.0  # iau time interval (to scale increments) in hours
        data["iau_inc_files"] = np.zeros(7)  # list of increment files
        data["iaufhrs"] = -1 * np.ones(
            7
        )  # forecast hours associated with increment files
        data["iau_filter_increments"] = False
        data[
            "sst_perturbation"
        ] = 0.0  # Sea surface temperature perturbation to climatology or nudging SST (default 0.0 K)

        self.data = data

    def read_namelist(self, input_nml_file):
        self.data["input_nml_file"] = input_nml_file

        with open(input_nml_file, "r") as f:
            nml_data = yaml.safe_load(f)

        gfs_physics_nml = nml_data["gfs_physics_nml"]
        for field in gfs_physics_nml.keys():
            self.data[field] = gfs_physics_nml[field]


class GFS_statein_type:
    def __init__(self, Model, IM):
        data = {}

        # --- level geopotential and pressures
        data["phii"] = np.zeros((IM, Model["levs"] + 1))
        data["prsi"] = np.zeros((IM, Model["levs"] + 1))
        data["prsik"] = np.zeros((IM, Model["levs"] + 1))

        # --- layer geopotential and pressures
        data["phil"] = np.zeros((IM, Model["levs"]))
        data["prsl"] = np.zeros((IM, Model["levs"]))
        data["prslk"] = np.zeros((IM, Model["levs"]))

        # --- shared radiation and physics variables
        data["vvl "] = np.zeros((IM, Model["levs"]))
        data["tgrs"] = np.zeros((IM, Model["levs"]))

        # stochastic physics SKEB variable
        data["diss_est"] = np.zeros((IM, Model["levs"]))

        # --- physics only variables
        data["pgr "] = np.zeros(IM)
        data["ugrs"] = np.zeros((IM, Model["levs"]))
        data["vgrs"] = np.zeros((IM, Model["levs"]))
        data["qgrs"] = np.zeros((IM, Model["levs"], Model["ntrac"]))

        # --- soil state variables - for soil SPPT - sfc-perts, mgehne
        data["smc"] = np.zeros((IM, Model["lsoil"]))
        data["stc"] = np.zeros((IM, Model["lsoil"]))
        data["slc"] = np.zeros((IM, Model["lsoil"]))

        # surface temperature from atmospheric prognostic state
        data["atm_ts"] = np.zeros(IM)

        data["dycore_hydrostatic"] = True

        data["nwat"] = 6

        self.data = data


class GFS_stateout_type:
    def __init__(self, Model, IM):
        data = {}

        data["gu0"] = np.zeros((IM, Model["levs"]))
        data["gv0"] = np.zeros((IM, Model["levs"]))
        data["gt0"] = np.zeros((IM, Model["levs"]))
        data["gq0"] = np.zeros((IM, Model["levs"], Model["ntrac"]))

        self.data = data


class GFS_sfcprop_type:
    def __init__(self, Model, IM):

        data = {}

        # --- physics and radiation
        data["slmsk"] = np.zeros(IM)
        data["oceanfrac"] = np.zeros(IM)
        data["landfrac"] = np.zeros(IM)
        data["lakefrac"] = np.zeros(IM)
        data["tsfc"] = np.zeros(IM)
        data["qsfc"] = np.zeros(IM)
        data["tsclim"] = np.zeros(IM)
        data["mldclim"] = np.zeros(IM)
        data["qfluxadj"] = np.zeros(IM)
        data["ts_som"] = np.zeros(IM)
        data["ts_clim_iano"] = np.zeros(IM)
        data["tml"] = np.zeros(IM)
        data["tml0"] = np.zeros(IM)
        data["mld"] = np.zeros(IM)
        data["mld0"] = np.zeros(IM)
        data["huml"] = np.zeros(IM)
        data["hvml"] = np.zeros(IM)
        data["tmoml"] = np.zeros(IM)
        data["tmoml0"] = np.zeros(IM)
        data["tsfco"] = np.zeros(IM)
        data["tsfcl"] = np.zeros(IM)
        data["tisfc"] = np.zeros(IM)
        data["snowd"] = np.zeros(IM)
        data["zorl"] = np.zeros(IM)
        data["zorlo"] = np.zeros(IM)
        data["zorll"] = np.zeros(IM)
        data["fice"] = np.zeros(IM)
        data["hprime"] = np.zeros((IM, Model["nmtvr"]))

        # --- In (radiation only)
        data["sncovr"] = np.zeros(IM)
        data["snoalb"] = np.zeros(IM)
        data["alvsf"] = np.zeros(IM)
        data["alnsf"] = np.zeros(IM)
        data["alvwf"] = np.zeros(IM)
        data["alnwf"] = np.zeros(IM)
        data["facsf"] = np.zeros(IM)
        data["facwf"] = np.zeros(IM)

        # --- physics surface props
        # --- In
        data["slope"] = np.zeros(IM)
        data["shdmin"] = np.zeros(IM)
        data["shdmax"] = np.zeros(IM)
        data["snoalb"] = np.zeros(IM)
        data["tg3"] = np.zeros(IM)
        data["vfrac"] = np.zeros(IM)
        data["vtype"] = np.zeros(IM)
        data["stype"] = np.zeros(IM)
        data["uustar"] = np.zeros(IM)
        data["oro"] = np.zeros(IM)
        data["oro_uf"] = np.zeros(IM)

        # --- In/Out
        data["hice"] = np.zeros(IM)
        data["weasd"] = np.zeros(IM)
        data["sncovr"] = np.zeros(IM)
        data["canopy"] = np.zeros(IM)
        data["ffmm"] = np.zeros(IM)
        data["ffhh"] = np.zeros(IM)
        data["f10m"] = np.zeros(IM)
        data["tprcp"] = np.zeros(IM)
        data["srflag"] = np.zeros(IM)
        data["slc"] = np.zeros((IM, Model["lsoil"]))
        data["smc"] = np.zeros((IM, Model["lsoil"]))
        data["stc"] = np.zeros((IM, Model["lsoil"]))

        # --- Out
        data["t2m"] = np.zeros(IM)
        data["q2m"] = np.zeros(IM)

        if Model["nstf_name"][0] > 0:
            data["tref"] = np.zeros(IM)
            data["z_c"] = np.zeros(IM)
            data["c_0"] = np.zeros(IM)
            data["c_d"] = np.zeros(IM)
            data["w_0"] = np.zeros(IM)
            data["w_d"] = np.zeros(IM)
            data["xt"] = np.zeros(IM)
            data["xs"] = np.zeros(IM)
            data["xu"] = np.zeros(IM)
            data["xv"] = np.zeros(IM)
            data["xz"] = np.zeros(IM)
            data["zm"] = np.zeros(IM)
            data["xtts"] = np.zeros(IM)
            data["xzts"] = np.zeros(IM)
            data["d_conv"] = np.zeros(IM)
            data["ifd"] = np.zeros(IM)
            data["dt_cool"] = np.zeros(IM)
            data["qrain"] = np.zeros(IM)

        if Model["lsm"] == Model["lsm_ruc"] or Model["lsm"] == Model["lsm_noahmp"]:
            data["raincprv"] = np.zeros(IM)
            data["rainncprv"] = np.zeros(IM)
            data["iceprv"] = np.zeros(IM)
            data["snowprv"] = np.zeros(IM)
            data["graupelprv"] = np.zeros(IM)

        if Model["lsm"] == Model["lsm_noahmp"]:
            data["snowxy"] = np.zeros(IM)
            data["tvxy"] = np.zeros(IM)
            data["tgxy"] = np.zeros(IM)
            data["canicexy"] = np.zeros(IM)
            data["canliqxy"] = np.zeros(IM)
            data["eahxy"] = np.zeros(IM)
            data["tahxy"] = np.zeros(IM)
            data["cmxy"] = np.zeros(IM)
            data["chxy"] = np.zeros(IM)
            data["fwetxy"] = np.zeros(IM)
            data["sneqvoxy"] = np.zeros(IM)
            data["alboldxy"] = np.zeros(IM)
            data["qsnowxy"] = np.zeros(IM)
            data["wslakexy"] = np.zeros(IM)
            data["zwtxy"] = np.zeros(IM)
            data["waxy"] = np.zeros(IM)
            data["wtxy"] = np.zeros(IM)
            data["lfmassxy"] = np.zeros(IM)
            data["rtmassxy"] = np.zeros(IM)
            data["stmassxy"] = np.zeros(IM)
            data["woodxy"] = np.zeros(IM)
            data["stblcpxy"] = np.zeros(IM)
            data["fastcpxy"] = np.zeros(IM)
            data["xsaixy"] = np.zeros(IM)
            data["xlaixy"] = np.zeros(IM)
            data["taussxy"] = np.zeros(IM)
            data["smcwtdxy"] = np.zeros(IM)
            data["deeprechxy"] = np.zeros(IM)
            data["rechxy"] = np.zeros(IM)

            data["snicexy"] = np.zeros((IM, 3))
            data["snliqxy"] = np.zeros((IM, 3))
            data["tsnoxy "] = np.zeros((IM, 3))
            data["smoiseq"] = np.zeros((IM, 4))
            data["zsnsoxy"] = np.zeros((IM, 7))

            data["draincprv"] = np.zeros(IM)
            data["drainncprv"] = np.zeros(IM)
            data["diceprv"] = np.zeros(IM)
            data["dsnowprv"] = np.zeros(IM)
            data["dgraupelprv"] = np.zeros(IM)

            self.data = data


class GFS_coupling_type:
    def __init__(self, Model, IM):

        data = {}

        # --- radiation out
        # --- physics in
        data["nirbmdi"] = np.zeros(IM)
        data["nirdfdi"] = np.zeros(IM)
        data["visbmdi"] = np.zeros(IM)
        data["visdfdi"] = np.zeros(IM)
        data["nirbmui"] = np.zeros(IM)
        data["nirdfui"] = np.zeros(IM)
        data["visbmui"] = np.zeros(IM)
        data["visdfui"] = np.zeros(IM)

        data["sfcdsw"] = np.zeros(IM)
        data["sfcnsw"] = np.zeros(IM)
        data["sfcdlw"] = np.zeros(IM)

        if Model["cplflx"] or Model["do_sppt"] or Model["cplchm"]:
            data["rain_cpl"] = np.zeros(IM)
            data["snow_cpl"] = np.zeros(IM)

        if Model["cplflx"] or Model["cplwav"]:
            # --- instantaneous quantities
            data["u10mi_cpl"] = np.zeros(IM)
            data["v10mi_cpl"] = np.zeros(IM)

        if Model["cplflx"]:
            # --- incoming quantities
            data["slimskin_cpl"] = np.zeros(IM)
            data["dusfcin_cpl"] = np.zeros(IM)
            data["dvsfcin_cpl"] = np.zeros(IM)
            data["dtsfcin_cpl"] = np.zeros(IM)
            data["dqsfcin_cpl"] = np.zeros(IM)
            data["ulwsfcin_cpl"] = np.zeros(IM)
            data["tseain_cpl"] = np.zeros(IM)
            data["tisfcin_cpl"] = np.zeros(IM)
            data["ficein_cpl"] = np.zeros(IM)
            data["hicein_cpl"] = np.zeros(IM)
            data["hsnoin_cpl"] = np.zeros(IM)

            # --- accumulated quantities
            data["dusfc_cpl"] = np.zeros(IM)
            data["dvsfc_cpl"] = np.zeros(IM)
            data["dtsfc_cpl"] = np.zeros(IM)
            data["dqsfc_cpl"] = np.zeros(IM)
            data["dlwsfc_cpl"] = np.zeros(IM)
            data["dswsfc_cpl"] = np.zeros(IM)
            data["dnirbm_cpl"] = np.zeros(IM)
            data["dnirdf_cpl"] = np.zeros(IM)
            data["dvisbm_cpl"] = np.zeros(IM)
            data["dvisdf_cpl"] = np.zeros(IM)
            data["nlwsfc_cpl"] = np.zeros(IM)
            data["nswsfc_cpl"] = np.zeros(IM)
            data["nnirbm_cpl"] = np.zeros(IM)
            data["nnirdf_cpl"] = np.zeros(IM)
            data["nvisbm_cpl"] = np.zeros(IM)
            data["nvisdf_cpl"] = np.zeros(IM)

            # --- instantaneous quantities
            data["dusfci_cpl"] = np.zeros(IM)
            data["dvsfci_cpl"] = np.zeros(IM)
            data["dtsfci_cpl"] = np.zeros(IM)
            data["dqsfci_cpl"] = np.zeros(IM)
            data["dlwsfci_cpl"] = np.zeros(IM)
            data["dswsfci_cpl"] = np.zeros(IM)
            data["dnirbmi_cpl"] = np.zeros(IM)
            data["dnirdfi_cpl"] = np.zeros(IM)
            data["dvisbmi_cpl"] = np.zeros(IM)
            data["dvisdfi_cpl"] = np.zeros(IM)
            data["nlwsfci_cpl"] = np.zeros(IM)
            data["nswsfci_cpl"] = np.zeros(IM)
            data["nnirbmi_cpl"] = np.zeros(IM)
            data["nnirdfi_cpl"] = np.zeros(IM)
            data["nvisbmi_cpl"] = np.zeros(IM)
            data["nvisdfi_cpl"] = np.zeros(IM)
            data["t2mi_cpl"] = np.zeros(IM)
            data["q2mi_cpl"] = np.zeros(IM)
            data["tsfci_cpl"] = np.zeros(IM)
            data["psurfi_cpl"] = np.zeros(IM)
            data["oro_cpl"] = np.zeros(IM)
            data["slmsk_cpl"] = np.zeros(IM)

        # -- cellular automata
        if Model["do_ca"]:
            data["tconvtend"] = np.zeros((IM, Model["levs"]))
            data["qconvtend"] = np.zeros((IM, Model["levs"]))
            data["uconvtend"] = np.zeros((IM, Model["levs"]))
            data["vconvtend"] = np.zeros((IM, Model["levs"]))
            data["cape"] = np.zeros(IM)
            data["ca_out"] = np.zeros(IM)
            data["ca_deep"] = np.zeros(IM)
            data["ca_turb"] = np.zeros(IM)
            data["ca_shal"] = np.zeros(IM)
            data["ca_rad"] = np.zeros(IM)
            data["ca_micro"] = np.zeros(IM)

        # -- GSDCHEM coupling options
        if Model["cplchm"]:
            # --- outgoing instantaneous quantities
            data["ushfsfci"] = np.zeros(IM)
            data["dkt"] = np.zeros((IM, Model["levs"]))
            data["dqdti"] = np.zeros((IM, Model["levs"]))
            # --- accumulated convective rainfall
            data["rainc_cpl"] = np.zeros(IM)

        # --- stochastic physics option
        if Model["do_sppt"]:
            data["sppt_wts"] = np.zeros((IM, Model["levs"]))

        # --- stochastic shum option
        if Model["do_shum"]:
            data["shum_wts"] = np.zeros((IM, Model["levs"]))

        # --- stochastic skeb option
        if Model["do_skeb"]:
            data["skebu_wts"] = np.zeros((IM, Model["levs"]))
            data["skebv_wts"] = np.zeros((IM, Model["levs"]))

        # --- stochastic physics option
        if Model["do_sfcperts"]:
            data["sfc_wts"] = np.zeros((IM, Model["nsfcpert"]))

        # --- needed for Thompson's aerosol option
        if Model["imp_physics"] == Model["imp_physics_thompson"] and Model["ltaerosol"]:
            data["nwfa2d"] = np.zeros(IM)
            data["nifa2d"] = np.zeros(IM)

        self.data = data
