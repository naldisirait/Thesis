import torch
import torch.nn as nn
import numpy as np
class Configuration:
    def __init__(self, config):
        """
        Class for configuration of the experiment
        """
        self.config = config
        self.model = None

    def get_learning_rate(self):
        return self.config['learning rate']
    
    def get_number_class(self):
        return self.config['number_class']
    
    def get_optimizer(self):
        optim_name = self.config['optimizer']
        if optim_name == "adam":
            lr = self.get_learning_rate()
            optimizer = torch.optim.Adam(self.model.parameters(),lr)
        elif optim_name == "..":
            pass
        return optimizer
    
    def get_epochs(self):
        return self.config['epochs']
    
    def get_criterion(self):
        criterion_name = self.config['criterion']
        if criterion_name == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif criterion_name == "MSE":
            pass
        return criterion
    
    def get_path_experiment(self):
        return self.config['path_experiment']
    
    def get_path_dataset(self):
        return self.config['path_dataset']
    
    def get_list_train_data(self):
        return self.config['train_data']
    
    def get_list_val_data(self):
        return self.config['val_data']
    
    def get_batch_size(self):
        return self.config['batch_size']
    
    def get_model_version(self):
        return self.config['model_version']
    
    def get_gpu_index(self):
        return self.config['gpu_index']
    
    def set_model(self, model):
        self.model = model

    def set_start_run_time(self, start_run_time):
        self.config['start_run_time'] = start_run_time

    def set_end_run_time(self, end_run_time):
        self.config['end_run_time'] = end_run_time

    def get_config(self):
        return self.config