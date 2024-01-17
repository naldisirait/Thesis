import numpy as np
import torch

def filter_error_shape(dataset : dict) -> dict:
    """
    Function to filter error data, because when cropping the data there might be a window through
    the data boundary.

    Args:
        dataset: a dictionary of our dataset
    
    """
    ir_array = dataset['IR array']
    idx_chosen = []
    idx_error = []
    for i in range(len(ir_array)):
        if ir_array[i].shape == (16,200,200):
            idx_chosen.append(i)
        else:
            idx_error.append(i)

    if len(idx_error) > 0:
        for key in dataset:
            dataset[key] = [dataset[key][i] for i in idx_chosen]

    return dataset

def intensity_classification(wind):
    """
    Function to classify the wind base on the tropical cyclone classification
    Args:
        wind: an array of the wind
    Return:
        classes: class based on the wind value
    """
    classes = []
    for w in (wind):
        if (w < 17.2):
            classes.append(0)
        elif (17.2 <= w < 24.5):
            classes.append(1)
        elif (24.5 <= w < 32.7):
            classes.append(2)
        elif (32.7 <= w < 41.5):
            classes.append(3)
        elif (41.5 <= w <51):
            classes.append(4)
        elif w >= 51:
            classes.append(5)
    return np.array(classes)

def data_processing(train_dataset: dict,val_dataset: dict):
    """
    function to process data into X_train, y_train, X_val, y_val
    Args:
        train_dataset: dictionary of the train dataset
        val_dataset: dictionary of the validation dataset

    """
    #filter error data
    train_dataset = filter_error_shape(train_dataset)
    val_dataset = filter_error_shape(val_dataset)
    
    #create tensor of infrared data train and  validation
    ir_train = np.array(train_dataset['IR array'])
    d1,d2,d3,d4 = ir_train.shape
    ir_train = torch.tensor(ir_train, dtype = torch.float32).reshape(d1*d2,1,200,200)

    #create wind class data train and validation
    wind_train = np.array(train_dataset['CMA wind'])
    d1,d2 =  wind_train.shape
    wind_train = np.reshape(wind_train, (d1*d2))
    wind_class_train = torch.tensor(intensity_classification(wind_train), dtype = torch.int64)

    #validation data
    ir_val = np.array(val_dataset['IR array'])
    d1,d2,d3,d4 = ir_val.shape
    ir_val = torch.tensor(ir_val, dtype = torch.float32).reshape(d1*d2,1,200,200)

    wind_val = np.array(val_dataset['CMA wind'])
    d1,d2 =  wind_val.shape
    wind_val = np.reshape(wind_val, (d1*d2))
    wind_class_val = torch.tensor(intensity_classification(wind_val), dtype = torch.int64)
    
    return ir_train,wind_class_train,ir_val,wind_class_val

