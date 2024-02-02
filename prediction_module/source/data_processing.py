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
    pass



def process_dataset(dataset):
    dataset = filter_error_shape(dataset)
