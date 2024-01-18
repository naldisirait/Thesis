import numpy as np
import pandas as pd
import torch 
import torch.nn as nn

from source.building_data_loader import create_data_loader
from source.evaluate_model import evaluate_model
from source.processing_data import proces_data
from source.ingesting_data import load_dataset
from source.train_model import train_model
from source.model_net import create_model
from utils.saving_experiment import save_experiments

def run_experiment():
    #1. Ingest Dataset
    load_dataset()

    #2. Data Processing
    proces_data()

    #3. Create data loader
    create_data_loader()

    #4. Build model
    create_model()

    #5. Train the model
    train_model()

    #6. Evaluate the model
    evaluate_model()

    #7. Save the experiment
    save_experiments()

    print("Successfully run the blueprint!")

if __name__ == "__main__":
    run_experiment()