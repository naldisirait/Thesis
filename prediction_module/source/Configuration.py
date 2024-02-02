class Configuration:
    def __init__(self, config):
        self.config = config

    def get_path_dataset(self):
        return self.config['path_dataset']
    
    def get_criterion(self):
        criterion = None
        return criterion

    def get_optimzer(self):
        optimizer = None
        return optimizer
    
    def get_epochs(self):
        return self.config['epochs']
    
    def get_batch_size(self):
        return self.config['batch_size']
    
    
