import numpy as np
import pandas as pd
import torch
import os
import yaml
import torch.nn as nn

from utils.utils_intensity import create_folder

def check_number_of_experiments(path: str) -> int:
    """
    Function to get the number of experiment that have been carried out.
    Args:
        path: path to the experiments
    Returns:
        len(items): number of experiments 

    """
    items = os.listdir(path)
    return len(items)

def save_config(config: dict, path: str):
    """
    function to save the config into yaml file
    Args:
        config: configuration of the experiment
        path: path to the file to be saved
    """

    # Write the dictionary to a YAML file
    with open(path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def save_acc_train_val(path: str, train_acc: list, val_acc: list):
    """
    Function to save the accuracy of model while training
    Args:
        path: path to the file, path should be in the current order experiment
        train_acc: model perfomanced on training datadata
        val_acc: model perfomanced on validation data 
    """
    df = pd.DataFrame({"Train Accuracy": train_acc,
                      "Validation Accuracy": val_acc})
    df.to_csv(f"{path}/model_accuracy.csv")

def save_model(path,model):
    """
    Function to save model
    Args:
        path: path to the model to be saved
        model: pytorch model
    """
    torch.save(model.state_dict(), path)

def save_experiment(path, config, model, train_acc, val_acc):
    """
    Function to save the experiment

    Args:
        path: path to experiments folder
        config: configuration of the model(lr, data, criterion etc)
        model: trained model in an experiments
        train_acc: a list of perfomanced model on train data per epochs
        val_acc: a list of perfomanced model on validation data per epochs

    """
    # check number of experiments and create a folder
    n_experiments = check_number_of_experiments(path)
    experiment_folder_name = f"experiment_{n_experiments}"
    create_folder(path, experiment_folder_name)

    #save the model and its perfomance(train_acc and val_acc)
    path_experiment_folder = f"{path}/{experiment_folder_name}"
    model_name = "model.pth"
    save_model(f"{path_experiment_folder}/{model_name}", model)
    save_acc_train_val(path_experiment_folder,train_acc,val_acc)

    

