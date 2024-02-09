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

def load_dataset(path,list_of_filename):
    for n,filename in enumerate(list_of_filename):
        if n == 0:
            dataset = open_pickle_dataset(f"{path}/{filename}")
        else:
            dataset = merge_dicts(dataset,open_pickle_dataset(f"{path}/{filename}"))
    return dataset