from physics.drivers import microphysics
import xarray as xr


def nc_to_dict(f):
    output = {}
    for varname, da in f.data_vars.items():
        output[varname] = da.to_numpy()
    return output


input_data = xr.open_dataset("c48_standard_tile_1.nc")
input_data = nc_to_dict(input_data)
output_data = microphysics.run(input_data, 79, 79, 225.0)
