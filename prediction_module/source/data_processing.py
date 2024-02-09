import torch
import numpy as np

def filter_error_shape(datasets):
    ir_array = datasets['IR array']
    idx_chosen = []
    idx_error = []
    for i in range(len(ir_array)):
        if ir_array[i].shape == (16,200,200):
            idx_chosen.append(i)
        else:
            idx_error.append(i)
    if len(idx_error) > 0:
        for key in datasets:
            datasets[key] = [datasets[key][i] for i in idx_chosen]        
    return datasets

def seperate_input_output(dataset):
        """
        Function to seperate past and future frame sequence, 
        input is a dataset from pickle 
        
        """
        X = [data_arr[0:8] for data_arr in dataset["IR array"]]
        y = [data_arr[8:16] for data_arr in dataset["IR array"]]
        
        X = torch.tensor(np.array(X), dtype = torch.float32).reshape(-1,8,1,200,200)
        y = torch.tensor(np.array(y), dtype = torch.float32).reshape(-1,8,1,200,200)
        
        return (X,y)

def process_dataset(dataset):
    dataset = filter_error_shape(dataset)
    X, y = seperate_input_output(dataset)
    return X,y
