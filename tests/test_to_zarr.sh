#!/bin/bash

set -e

pytest serial_convert/

out_dir=$(mktemp -d)
python serial_convert/serial_to_zarr.py \
    tests/test_data \
    serial_convert/turb_parameter_metadata.yaml \
    $out_dir

python tests/test_serial_to_zarr.py tests/test_data $out_dir

