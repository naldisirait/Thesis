import os
import pandas as pd
import numpy as np
import json
import logging

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
    """
    Write a dictionary to a JSON file.
    Args:
        - data (dict): The dictionary to be written to the JSON file.
        - file_path (str): The path to the JSON file.
    Returns:
        None
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


