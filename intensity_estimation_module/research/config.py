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
    
    def set_model(self, model):
        self.model = model