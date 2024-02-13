import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import torch

def create_folder(path:str, folder_name: str):
    """
    Function to create folder

    Args:
        path: path to folder
        folder_name: name of the folder
    """
    # Create the folder
    fullpath = f"{path}/{folder_name}"
    os.makedirs(fullpath, exist_ok=True)
    logging.info(f"Folder '{folder_name}' created at: {os.path.abspath(path)}")

def read_json_file(file_path: str) -> dict:
    """
    Read a JSON file and return its content as a dictionary.
    Args:
        - file_path (str): The path to the JSON file.
    Returns:
    - dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(data: dict, file_path: str):
    file_path = f"{file_path}/config.json"
    """
    Write a dictionary to a JSON file.
    Args:
        - data (dict): The dictionary to be written to the JSON file.
        - file_path (str): The path to the JSON file.
    Returns:
        None
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def get_current_time():
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the current time as a string
    formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_time

def load_pretrained_model(model, pre_trained_model: str):
    """
    Load weights from pretrained model
    Args:
        - model : PyTorch Pretrained model
        - filename: file path to the pretrained_model
    """
    model = model.load_state_dict(torch.load(pre_trained_model))
    return model

def save_model(model, path, output_name):
    #Save model
    torch.save(model.state_dict(),f"{path}/{output_name}.pt")