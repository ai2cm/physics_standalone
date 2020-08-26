#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np

OUT_VARS = ["dv", "du", "tdt", "rtg", "kpbl", "dusfc", "dvsfc", "dtsfc", "dqsfc", "hpbl"]

def run(in_dict):
    """run function"""

    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = np.zeros(1, dtype=np.float64)

    return out_dict


