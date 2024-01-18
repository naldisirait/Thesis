from source.data_ingest import load_dataset
from source.data_processing import data_processing
from source.building_data_loader import create_data_loader
from source.model_net import create_model_efficient_net
from source.train_model import train_net
from utils.saving_experiments import save_experiment
from utils.utils_intensity import read_json_file
from research.config import Configuration
from utils.utils_intensity import get_current_time

import torch
import torch.nn as nn

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 100, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

def run_experiment():
    #load the configuration for this experiment
    config = read_json_file("config.json")
    config_class = Configuration(config)
    
    #get start run time and save into the config experiment
    config_class.set_start_run_time(get_current_time())

    #0. set the computation device
    gpu_idx = config_class.get_gpu_index()
    device = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"0. Using device: {device}")

    path_dataset = config_class.get_path_dataset()
    file_train = config_class.get_list_train_data()
    file_val = config_class.get_list_val_data()

    #1. load dataset
    train_dataset = load_dataset(path_dataset = path_dataset, data_name=file_train)
    val_dataset = load_dataset(path_dataset= path_dataset, data_name= file_val)

    print("1. Ingesting dataset DONE.")

    #2. processing dataset
    X_train, y_train, X_val, y_val = data_processing(train_dataset=train_dataset,val_dataset=val_dataset)
    X_train, y_train, X_val, y_val = X_train[0:100], y_train[0:100], X_val[0:100], y_val[0:100]
    print("2. Processing data DONE")

    #3. Create custom data loader
    batch_size = config_class.get_batch_size()
    train_loader = create_data_loader(X = X_train, y = y_train, batch_size = batch_size, shuffle=True)
    val_loader = create_data_loader(X = X_val, y = y_val, batch_size = batch_size, shuffle = False)
    print("3. Creating Data Loader DONE")

    #4. create the model
    version = config_class.get_model_version()
    num_classes = config_class.get_number_class()
    model = create_model_efficient_net(version=version, num_classes=num_classes, device=device, phi_values= phi_values)
    config_class.set_model(model)

    optimizer = config_class.get_optimizer()
    criterion = config_class.get_criterion()
    num_epochs = config_class.get_epochs()
    experiment_path = config_class.get_path_experiment()

    print("4. Creating Model DONE")
    #5. train the model
    model,best_model, train_accuracy,val_accuracy = train_net(model, 
                                                  num_epochs=num_epochs,
                                                  train_loader= train_loader,
                                                  valid_loader= val_loader,
                                                  criterion= criterion,
                                                  optimizer= optimizer,
                                                  device=device)
    
    print("5. Training model DONE")

    #get end run time and save into the config experiment
    config_class.set_end_run_time(get_current_time())

    save_experiment(path = experiment_path, 
                    config = config_class.get_config(), 
                    model = model,
                    best_model = best_model, 
                    train_acc=train_accuracy, 
                    val_acc= val_accuracy)
    
    print("6. Succesfully saving experiment.")

if __name__ == "__main__":
    run_experiment()