import numpy as np
import xarray as xr

def make_nc2D(
        data_array : np.ndarray,
        lat : np.ndarray,
        lon : np.ndarray,
        parameter : str,
        path_output : str,
        output_name: str):
    
    """
    Function to dump an array data into nc file

    Args:
        data_array : 2 dimensional array 
        lat : a vector of the latitude array
        lon : a vector of the longitude array
        parameter : a name representation for the data,
        path_output : output path,
        output_name: output nc filename):
    
    """
    #encode = {parameter: {"zlib":True, "complevel":9}}
    dxr = xr.Dataset(
    {f"{parameter}": (("latitude", "longitude"), data_array)},
    coords={
        "latitude": lat,
        "longitude": lon,
        })
    dxr.to_netcdf(f"{path_output}{output_name}.nc") #,encoding = encode)