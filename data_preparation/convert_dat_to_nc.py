import numpy as np
import xarray as xr
import yaml
import os
import logging
from scipy import interpolate
import time
from utils import make_nc2D, read_from_yaml

def convert_dat_to_nc(path_dat,path_nc,resolution):
    #create a list of all file name from path_dat
    files = os.listdir(path_dat)
    
    #list all files nc, so we dont overwrite the same file
    files_nc = os.listdir(path_nc)
    
    #select file .dat only
    dat_files  = [file for file in files if file[-4:] == '.dat']

    dat_files = [file for file in dat_files[:] if file [0:-4]+".nc" not in files_nc]

    for n,file in enumerate(dat_files):
        #open dat file
        tir_raw = np.fromfile(path_dat+file, dtype = "float32")
        tir_raw = np.reshape(tir_raw, (6000,6000))
        
        #Coordinate raw data
        lon_raw = np.linspace(85,205,6000) #define lon
        lat_raw = np.linspace(60,-60,6000) # define lat
        
        interval = resolution
        
        lat_modified = np.arange(60,-60-interval, -interval)
        lon_modified = np.arange(85,205+interval, interval)

        interp_func = interpolate.interp2d(lat_raw, lon_raw, tir_raw, kind='linear')
        
        # Extrapolate data points to the 10km resolution
        tir_modified = interp_func(lat_modified, lon_modified)
        tir_modified = np.flip(tir_modified, axis=1)
        
        # convert to nc
        make_nc2D(tir_modified,lat_modified,lon_modified,"TIR01",path_nc,file[0:-4])

def run_converting():
    try:
        config = read_from_yaml("C:/Users/62812/Documents/Kuliah/Thesis/Thesis/data_preparation/config.yaml")
        
        #get the configuration when doing converting
        resolution = config['resolution']
        path_dat = config['path_dat']
        path_nc = config['path_nc']

        #run script data preperation to get tc dataset
        start_time =time.time()
        resolution = 0.1
        convert_dat_to_nc(path_dat,path_nc,resolution)
        end_time = time.time()
        runtime = end_time - start_time

        print(f"Runtime: {runtime} seconds")

    except Exception as e:
        logging.error(e)
        raise e
    
if __name__ == "__main__":
    run_converting()