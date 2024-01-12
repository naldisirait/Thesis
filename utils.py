import pickle

def open_pickle_dataset(file_name):
    
    #Load the variable from the pickle file
    with open(file_name, "rb") as file:
        dataset = pickle.load(file)
        
    return dataset