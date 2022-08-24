import os
import numpy as np
from gt4py import gtscript

# prefix = "cloud_mp"
prefix = "Microph"
### SERIALIZATION ###

IS_DOCKER = (os.getenv("IS_DOCKER") == "True") if ("IS_DOCKER" in os.environ) else False

# Path of serialbox directory
if IS_DOCKER:
    SERIALBOX_DIR = "/usr/local/serialbox"
else:
    SERIALBOX_DIR = "/project/s1053/install/serialbox/gnu"


# Names of the input variables
IN_VARS = [
    "iie",
    "kke",
    "kbot",
    "qv",
    "ql",
    "qr",
    "qg",
    "qa",
    "qn",
    "pt",
    "uin",
    "vin",
    "dz",
    "delp",
    "area",
    "dt_in",
    "land",
    "seconds",
    "p",
    "lradar",
    "reset",
    "qi",
    "qs",
    "qv_dt",
    "ql_dt",
    "qr_dt",
    "qi_dt",
    "qs_dt",
    "qg_dt",
    "qa_dt",
    "pt_dt",
    "w",
    "udt",
    "vdt",
    "rain",
    "snow",
    "ice",
    "graupel",
    "refl_10cm",
]

IN_VARS_PACE = [
    "mph_qv1",
    "mph_ql1",
    "mph_qr1",
    "mph_qi1",
    "mph_qs1",
    "mph_qg1",
    "mph_qa1",
    "mph_qn1",
    "mph_qv_dt",
    "mph_ql_dt",
    "mph_qr_dt",
    "mph_qi_dt",
    "mph_qs_dt",
    "mph_qg_dt",
    "mph_qa_dt",
    "mph_pt_dt",
    "mph_pt",
    "mph_w",
    "mph_uin",
    "mph_vin",
    "mph_udt",
    "mph_vdt",
    "mph_dz",
    "mph_delp",
    "mph_area",
    "mph_dtp_in",
    "mph_land",
    "mph_rain0",
    "mph_snow0",
    "mph_ice0",
    "mph_graupel0",
    "mph_im",
    "mph_levs",
    "mph_seconds",
    "mph_p123",
    "mph_lradar",
    "mph_refl",
    "mph_reset",
]

pace_dict_lookup = {}
pace_dict_lookup["mph_qv1"] = "qv"
pace_dict_lookup["mph_ql1"] = "ql"
pace_dict_lookup["mph_qr1"] = "qr"
pace_dict_lookup["mph_qi1"] = "qi"
pace_dict_lookup["mph_qs1"] = "qs"
pace_dict_lookup["mph_qg1"] = "qg"
pace_dict_lookup["mph_qa1"] = "qa"
pace_dict_lookup["mph_qn1"] = "qn"
pace_dict_lookup["mph_qv_dt"] = "qv_dt"
pace_dict_lookup["mph_ql_dt"] = "ql_dt"
pace_dict_lookup["mph_qr_dt"] = "qr_dt"
pace_dict_lookup["mph_qi_dt"] = "qi_dt"
pace_dict_lookup["mph_qs_dt"] = "qs_dt"
pace_dict_lookup["mph_qg_dt"] = "qg_dt"
pace_dict_lookup["mph_qa_dt"] = "qa_dt"
pace_dict_lookup["mph_pt_dt"] = "pt_dt"
pace_dict_lookup["mph_pt"] = "pt"
pace_dict_lookup["mph_w"] = "w"
pace_dict_lookup["mph_uin"] = "uin"
pace_dict_lookup["mph_vin"] = "vin"
pace_dict_lookup["mph_udt"] = "udt"
pace_dict_lookup["mph_vdt"] = "vdt"
pace_dict_lookup["mph_dz"] = "dz"
pace_dict_lookup["mph_delp"] = "delp"
pace_dict_lookup["mph_area"] = "area"
pace_dict_lookup["mph_dtp_in"] = "dt_in"
pace_dict_lookup["mph_land"] = "land"
pace_dict_lookup["mph_rain0"] = "rain"
pace_dict_lookup["mph_snow0"] = "snow"
pace_dict_lookup["mph_ice0"] = "ice"
pace_dict_lookup["mph_graupel0"] = "graupel"
pace_dict_lookup["mph_im"] = "iie"
pace_dict_lookup["mph_levs"] = "kke"
pace_dict_lookup["mph_levs"] = "kbot"
pace_dict_lookup["mph_seconds"] = "seconds"
pace_dict_lookup["mph_p123"] = "p"
pace_dict_lookup["mph_lradar"] = "lradar"
pace_dict_lookup["mph_refl"] = "refl_10cm"
pace_dict_lookup["mph_reset"] = "reset"

# Names of the output variables
OUT_VARS = [
    "qi",
    "qs",
    "qv_dt",
    "ql_dt",
    "qr_dt",
    "qi_dt",
    "qs_dt",
    "qg_dt",
    "qa_dt",
    "pt_dt",
    "w",
    "udt",
    "vdt",
    "rain",
    "snow",
    "ice",
    "graupel",
    "refl_10cm",
]

OUT_VARS_pace = [
    "mph_qi1",
    "mph_qs1",
    "mph_qv_dt",
    "mph_ql_dt",
    "mph_qr_dt",
    "mph_qi_dt",
    "mph_qs_dt",
    "mph_qg_dt",
    "mph_qa_dt",
    "mph_pt_dt",
    "mph_w",
    "mph_udt",
    "mph_vdt",
    "mph_rain0",
    "mph_snow0",
    "mph_ice0",
    "mph_graupel0",
    "mph_refl",
]
# Names of the integer and boolean variables
INT_VARS = ["iie", "kke", "kbot", "seconds", "lradar", "reset"]

# Names of the float variables
FLT_VARS = [
    "qv",
    "ql",
    "qr",
    "qg",
    "qa",
    "qn",
    "pt",
    "uin",
    "vin",
    "dz",
    "delp",
    "area",
    "dt_in",
    "land",
    "p",
    "qi",
    "qs",
    "qv_dt",
    "ql_dt",
    "qr_dt",
    "qi_dt",
    "qs_dt",
    "qg_dt",
    "qa_dt",
    "pt_dt",
    "w",
    "udt",
    "vdt",
    "rain",
    "snow",
    "ice",
    "graupel",
    "refl_10cm",
]


### UTILITY ###

# Predefined types of variables and fields
DTYPE_INT = np.int32
DTYPE_FLT = np.float64
FIELD_INT = gtscript.Field[DTYPE_INT]
FIELD_FLT = gtscript.Field[DTYPE_FLT]

# GT4PY parameters
BACKEND = str(os.getenv("BACKEND")) if ("BACKEND" in os.environ) else "gtx86"
REBUILD = (os.getenv("REBUILD") == "True") if ("REBUILD" in os.environ) else False
DEBUG_MODE = (
    (os.getenv("DEBUG_MODE") == "True") if ("DEBUG_MODE" in os.environ) else False
)
DEFAULT_ORIGIN = (0, 0, 0)

# Stencils mode
STENCILS = (
    str(os.getenv("STENCILS"))
    if (("STENCILS" in os.environ) and USE_GT4PY)
    else "normal"
)
