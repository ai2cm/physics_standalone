import zarr
import rechunker
import argparse
import logging
import numpy as np
from functools import partial
from tempfile import TemporaryDirectory
from pathlib import Path
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)


def _in_mb(nbytes_per_item, size):
    return size * nbytes_per_item // 1024**2

def get_target_chunk(data, target_size_mb):

    chunks = []
    shape = data.shape
    itemsize = data.dtype.itemsize
    get_mb = partial(_in_mb, itemsize)

    curr_size = 1
    for dim_size in shape[::-1]:
        
        if get_mb(curr_size) >= target_size_mb:
            chunks.insert(0, 1)
        else:
            for i in range(1, dim_size + 1):
                if get_mb(curr_size * i) >= target_size_mb:
                    chunks.insert(0, i)
                    curr_size *= i
                    break
            else:
                chunks.insert(0, dim_size)
                curr_size *= dim_size

    return chunks

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_zarr")
    parser.add_argument("output_zarr")
    parser.add_argument("--chunksize_mb", required=False, default=5, type=int)

    args = parser.parse_args()

    max_mem = "256MB"

    logger.info(f"Rechunking zarr at path: {args.input_zarr}")
    dataset = zarr.open(args.input_zarr)
    with TemporaryDirectory() as tmpdir:
        filename = Path(args.input_zarr).name
        tmp_zarr = Path(tmpdir, filename).as_posix()
        chunks = {var: get_target_chunk(dataset[var], args.chunksize_mb) for var in dataset}
        plan = rechunker.rechunk(dataset, chunks, max_mem, args.output_zarr, tmp_zarr)
        with ProgressBar():
            plan.execute()
