import numpy as np
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
import datetime
import warnings
warnings.filterwarnings('ignore')

def open_nc(path,date):
    #open nc file using xarray
    tir = xr.open_dataset(path+f"TIR01_{date}.nc")
    tir_val = tir['TIR01'].values
    return tir_val

#function to calculate the distance between two points
def distance_point(t1,t2):
    #define lat lon from each points
    lat1,lon1 = t1[0],t1[1]
    lat2,lon2 = t2[0],t2[1]
    
    #calculate distance between 2 points
    d = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)
    return d

#function to conver string timestamp to unix timestamp
def iso_time_to_unix(iso_time):
    #Convert the string to a datetime object
    str_timestamp = datetime.datetime.strptime(iso_time, "%Y-%m-%d %H:%M:%S")
    
    #convert datetime to Unix timestamp
    unix_timestamp = int(str_timestamp.timestamp())
    return unix_timestamp

def generate_sequence_date(center_date, hours, interval):
    #function to create sequence of dates from center date chosen
    period = int(hours/interval)
    forward_dates = pd.date_range(start=center_date, periods=period+1, freq = f'{interval}H')
    backward_dates = pd.date_range(start=center_date, periods=period+1, freq=f'-{interval}H')
    backward_dates = backward_dates[:-1]
    dates = backward_dates[::-1].append(forward_dates[1:])
    return [str(i) for i in dates]

def create_date_from_dat(path):
    #list all the files in a path
    files = os.listdir(path)
    
    #pick only .dat data
    dat_files  = [file for file in files if file[-3:] == '.nc']
    
    #take substring as date from dat filename
    dates_dat = [f"{i[6:10]}-{i[10:12]}-{i[12:14]} {i[14:16]}:{i[16:18]}:00" for i in dat_files]
    return dates_dat

def check_gen_dates_in_dates_dat(dates_from_dat,gen_dates):
    #loop through the generated  dates and check if the date is in the list of data downloaded
    for i in gen_dates:
        if (i not in dates_from_dat):
            return False
    # if all the date in the list then we return True
    return True

#fungsi untuk mencari index titik
def find_idx(base,goal):
    idx = np.argmin(abs(base-goal))
    return idx

def crop_tc(raw_array,lat_center,lon_center,len_side):
    
    #Koordinat data
    interval = 0.1
    lat_10km = np.arange(60,-60-interval, -interval)
    lon_10km = np.arange(85,205+interval, interval)
    
    s = len_side
    
    #find index center latitude and longitude of center date
    idx_lat = find_idx(lat_10km,lat_center)
    idx_lon = find_idx(lon_10km,lon_center)
    
    #slice the raw data using index lat,lon, and length side
    raw_crop = raw_array[idx_lat-s:idx_lat+s,idx_lon-s:idx_lon+s]
    return raw_crop,lat_10km[idx_lat-s:idx_lat+s],lon_10km[idx_lon-s:idx_lon+s]

def remove_non_numeric(string):
    output = ""
    for char in string:
        if char.isdigit():
            output += char
    return output
                            
def create_sequence_array_and_fetaures_val(df_bestrack,path_dat,gen_dates,lat_center,lon_center,len_side):
    #remove all non string in the date so we can open the dat data using the generated dates
    format_dates_dat = []
    for date in gen_dates:
        format_dates_dat.append(remove_non_numeric(date[0:16]))
    
    #define variable to stored features values and IR arrays
    seq_arrays, tc_names, tc_dates, wmo_lats, wmo_lons, wmo_winds, cma_lats, cma_lons, cma_winds = [],[],[],[],[],[],[],[],[]
    
    for n,date in enumerate(format_dates_dat):
        #open raw data using path and date
        raw_array = open_nc(path_dat,date)
    
        #crop data based on lat and lot center date
        raw_crop, IR_lat_crop, IR_lon_crop = crop_tc(raw_array,lat_center,lon_center,len_side)

        #extract all the features values on the date
        feature_val = tuple(df_bestrack[df_bestrack['ISO_TIME'] == gen_dates[n]].values[0][5:-1])
        tc_name, tc_date, wmo_lat, wmo_lon, wmo_wind, cma_lat, cma_lon, cma_wind = feature_val
        
        #add the raw data to list of sequence array
        seq_arrays.append(raw_crop); tc_names.append(tc_name); tc_dates.append(tc_date); wmo_lats.append(wmo_lat)
        wmo_lons.append(wmo_lon); wmo_winds.append(wmo_wind); cma_lats.append(cma_lat); cma_lons.append(cma_lon); cma_winds.append(cma_wind)
        
    #convert to array    
    seq_array = np.array(seq_arrays) 
    
    return (seq_array,IR_lat_crop,IR_lon_crop,tc_names, tc_dates, wmo_lats, wmo_lons, wmo_winds, cma_lats, cma_lons, cma_winds)

def create_dataset_tc(df_bestrack,path_dat,hours,time_interval,len_side):
    
    #define dictioanary to get all the values from IR and betstrack dataset
    tc_dataset = {'IR array': [], 'IR lat': [], 'IR lon': [], 'Center Dates': [], 'Array Dates':[],
               'TC name': [], 'Center lat': [], 'Center lon': [], 'WMO wind': [], 'Center CMA lat':[],
               'Center CMA lon': [], 'CMA wind': []}
    
    #create dates from downloaded data
    dates_dat = create_date_from_dat(path_dat)
    
    #define tc date 
    tc_date = df_bestrack['ISO_TIME']
    
    for center_date in tc_date:
        #generate backwards and forward dates
        gen_dates = generate_sequence_date(center_date,hours,time_interval)
        
        #check if the center date is possible
        if check_gen_dates_in_dates_dat(dates_dat,gen_dates):
            #get lat and lon from the center date for defining the center box when cropping IR
            lat_center = df_bestrack[df_bestrack['ISO_TIME'] == center_date]['LAT'].values[0]
            lon_center = df_bestrack[df_bestrack['ISO_TIME'] == center_date]['LON'].values[0]
            
            #Create sequence array based on generated dates
            features_values_sequence = create_sequence_array_and_fetaures_val(df_bestrack,path_dat,gen_dates,lat_center,lon_center,len_side)
            seq_array,IR_lat_crop,IR_lon_crop,tc_names, tc_dates, wmo_lats, wmo_lons, wmo_winds, cma_lats, cma_lons, cma_winds = features_values_sequence
            
            #append all the features values into dictionary
            tc_dataset['IR array'].append(seq_array); tc_dataset['IR lat'].append(IR_lat_crop);tc_dataset['IR lon'].append(IR_lon_crop)
            tc_dataset['Center Dates'].append(center_date); tc_dataset['Array Dates'].append(tc_dates); tc_dataset['TC name'].append(tc_names)
            tc_dataset['Center lat'].append(wmo_lats); tc_dataset['Center lon'].append(wmo_lons); tc_dataset['WMO wind'].append(wmo_winds)
            tc_dataset['Center CMA lat'].append(cma_lats); tc_dataset['Center CMA lon'].append(cma_lons) ; tc_dataset['CMA wind'].append(cma_winds)
            
    return tc_dataset

#function to filter which sequence data have intensity (wind) in every frame
def filter_dataset_by_cma_wind(cma_wind):
    idx_incomplete = []
    for n,seq_wind in enumerate(cma_wind):
        fact = any(wind == -1.0 for wind in seq_wind)
        if fact:
            idx_incomplete.append(n)
    return idx_incomplete

def check_empty_sequence(tc_array):
    idx_empty = []
    for n,arr in enumerate(tc_array):
        if len(arr) == 0:
            idx_empty.append(n)
    return idx_empty

def filter_dataset_by_emptiness_and_incomplete_wind(tc_dataset,idx_empty_val,idx_wind_incomplete):
    indices_to_remove = list(set(idx_empty_val) | set(idx_wind_incomplete))
    for key in tc_dataset:
        tc_dataset[key] = [value for i,value in enumerate(tc_dataset[key]) if i not in indices_to_remove]
    return tc_dataset

def get_dataset(path_dat,df_bestrack,hours,len_side,time_interval):
    #Create tc dataset
    TC_datasets =  create_dataset_tc(df_bestrack,path_dat,hours,time_interval,len_side)
    
    #filter dataset
    """check empty sequence data, occurs because of the tropical cyclone happens near the border satellit area, 
    so when trying to slice the extended bounding box, the data is empty because the satellite didnt capture the extended area."""
    
    idx_empty_val = check_empty_sequence(TC_datasets['IR lat'])
    idx_wind_complete = filter_dataset_by_cma_wind(TC_datasets['CMA wind'])
    TC_datasets = filter_dataset_by_emptiness_and_incomplete_wind(TC_datasets,idx_empty_val,idx_wind_complete)

    return TC_datasets