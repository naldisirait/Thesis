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
    
    def get_learning_rate(self):
        return self.config['get_learning_rate']
    
    def get_epochs(self):
        return self.config['epochs']
    
    def get_batch_size(self):
        return self.config['batch_size']
    
    def get_list_of_train_data(self):
        return self.config['list_of_train_data']
    
    def get_list_of_validation_data(self):
        return self.config['list_of_validation_data']
    
    def get_pre_trained_model(self):
        return self.config['pre_trained_model']
    
    def get_gpu_index(self):
        return self.config['gpu_index']
    
