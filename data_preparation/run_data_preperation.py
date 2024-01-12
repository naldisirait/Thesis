import numpy as np
import pandas as pd
import pickle
import time
import os
import yaml
import xarray as xr

import building_dataset
from utils_preperation import make_nc2D, read_from_yaml



#tahun di ganti sesuai data tahun yang ingin diolah
tahun = '2021'
path_nc = "C:/Users/62812/Documents/Kuliah/Semester 3/Tesis/Data/Himawari/nc/2021 test/"

#Koordinat data raw
lon_base = np.linspace(85,205,6000) #define lon
lat_base = np.linspace(60,-60,6000) # define lat


def filter_df(path: str) -> pd.DataFrame:
    #open bestrack data
    bt_16_23 = pd.read_csv(path)
    #buat column tahun di data
    years = [i[0:4] for i in  bt_16_23['ISO_TIME']]
    bt_16_23['Tahun'] = years

    #ambil data tc di spesifik tahun saja
    bt_tahun =  bt_16_23[bt_16_23['Tahun'] == tahun]
    bt_tahun = bt_tahun.drop_duplicates(subset='ISO_TIME', keep="first")
    return bt_tahun


def run_building_dataset():
    config = read_from_yaml("C:/Users/62812/Documents/Kuliah/Thesis/Thesis/data_preparation/config.yaml")
    #set constants value
    tahun = config['tahun']
    hours = config['hours']
    len_side = config['len_side']
    time_interval = config['time_interval']
    path_bt = confi['path_besttrack']
    bt_tahun = filter_df(path_bt)

    #run script data preperation to get tc dataset
    start_time =time.time()

    TC_datasets = building_dataset.get_dataset(path_nc,bt_tahun,hours,len_side,time_interval)

    # File path to save the serialized variable
    file_name = f"TC Dataset {tahun} test {time_interval}.pkl"
    # Dump the variable to a file using pickle
    with open(file_name, "wb") as file:
        pickle.dump(TC_datasets, file)

    # Calculate the runtime
    end_time = time.time()
    runtime = end_time - start_time

    print(f"Runtime: {runtime} seconds")