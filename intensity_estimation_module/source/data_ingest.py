import numpy as np
import pickle

def open_pickle_dataset(file_name: str) -> dict:
    
    #Load the variable from the pickle file
    with open(file_name, "rb") as file:
        dataset = pickle.load(file)
        
    return dataset

def merge_dicts(dict1: dict, dict2: dict) -> dict:
    merged_dict = {}
    for key in dict1:
        if key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
        else:
            merged_dict[key] = dict1[key]
    
    return merged_dict

def load_dataset(path_dataset: str, data_name: list) -> dict:
    """
    Function to load dataset
    Args:
        path_dataset: path to the dataset
        data_name: list of filename
    
    """
    if len(data_name) > 1:
        for n in range(len(data_name)):
            if n == 0:
                dataset = open_pickle_dataset(f"{path_dataset}/{data_name[n]}.pkl")
            else:
                data_next = open_pickle_dataset(f"{path_dataset}/{data_name[n]}.pkl")
                dataset = merge_dicts(dataset,data_next)
    else:
        dataset =  open_pickle_dataset(f"{path_dataset}/{data_name[0]}.pkl")
    return dataset

