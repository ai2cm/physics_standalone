#!/usr/bin/env python3

import numpy as np

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

def run(in_dict):

    # TODO - implement sea-ice model
    return {key: in_dict.get(key, None) for key in OUT_VARS}