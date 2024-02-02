import pickle

def open_pickle_dataset(file_name):
    
    #Load the variable from the pickle file
    with open(file_name, "rb") as file:
        dataset = pickle.load(file)
        
    return dataset

def merge_dicts(dict1, dict2):
    merged_dict = {}
    
    for key in dict1:
        if key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
        else:
            merged_dict[key] = dict1[key]
    return merged_dict

def load_train_dataset():
    dataset_2018 = open_pickle_dataset("../Data Tesis/TC Dataset 2018.pkl")
    dataset_2019 = open_pickle_dataset("../Data Tesis/TC Dataset 2019.pkl")
    dataset_2020 = open_pickle_dataset("../Data Tesis/TC Dataset 2020.pkl")
    dataset_2021 = open_pickle_dataset("../Data Tesis/TC Dataset 2021.pkl")
    
    dataset = merge_dicts(dataset_2018, dataset_2019)
    dataset = merge_dicts(dataset, dataset_2020)
    dataset = merge_dicts(dataset, dataset_2021)
    
    dataset = merge_dicts(dataset_2020, dataset_2021)
    dataset = filter_error_shape(dataset)
    
    return dataset

def load_dataset():
    pass