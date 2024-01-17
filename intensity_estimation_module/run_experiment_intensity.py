from source.data_ingest import load_dataset
from source.data_processing import data_processing
from source.building_data_loader import create_data_loader
from source.model_net import create_model_efficient_net
from source.train_model import train_net
from utils.saving_experiments import save_experiment
from utils.utils_intensity import read_json_file
from research.config import Configuration

import torch
import torch.nn as nn

def run_experiment():
    #0. set the computation variable
    gpu_idx = 1
    device = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")

    path_dataset = "C:/Users/62812/Documents/Kuliah/Thesis/data"
    file_train = ["TC Dataset 2018", "TC Dataset 2019", "TC Dataset 2020", "TC Dataset 2021"]
    file_val = ["TC Dataset 2021 validation"]

    #1. load dataset
    train_dataset = load_dataset(path_dataset = path_dataset, data_name=file_train)
    val_dataset = load_dataset(path_dataset=path_dataset, data_name= file_val)

    #2. processing dataset
    X_train, y_train, X_val, y_val = data_processing(train_dataset=train_dataset,val_dataset=val_dataset)

    #3. Create custom data loader
    batch_size = 64
    train_loader = create_data_loader(X = X_train, y = y_train, batch_size = batch_size, shuffle=True)
    val_loader = create_data_loader(X = X_val, y = y_val, batch_size = batch_size, shuffle = False)

    #4. create the model
    version = "b0"
    num_classes = 6
    model = create_model_efficient_net(version=version, num_classes=num_classes)
    config = read_json_file("config.json")
    config_class = Configuration(config)
    config_class.set_model(model)

    optimizer = config_class.get_optimizer()
    criterion = config_class.get_criterion()
    num_epochs = config_class.get_epochs()
    experiment_path = config_class.get_path_experiment()

    #5. train the model
    model,train_accuracy,val_accuracy = train_net(model, 
                                                  num_epochs=num_epochs,
                                                  train_loader= train_loader,
                                                  valid_loader= val_loader,
                                                  criterion= criterion,
                                                  optimizer= optimizer,
                                                  device=device)
    
    save_experiment(path = experiment_path, 
                    config = config, 
                    model = model, 
                    train_acc=train_accuracy, 
                    val_acc= val_accuracy)

if __name__ == "__main__":
    run_experiment()