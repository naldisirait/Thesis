from source.ingest_data import load_dataset
from source.Configuration import Configuration
from source.data_processing import process_dataset
from source.building_data_loader import create_data_loader
from source.model import create_model_prediction
from source.train_model import train_model

def run_experiment(config):
    #Load dataset
    config_class = Configuration(config)
    dataset = load_dataset(config_class.path_datasets())
    X_train, y_train, X_val, y_val = process_dataset(dataset)
    data_train_loader = create_data_loader(X_train, y_train)
    data_val_loader = create_data_loader(X_val, y_val)
    model = create_model_prediction(config_class)
    model, train_loss, eval_loss = train_model(config_class, model, data_train_loader, data_val_loader)

if __name__ == "__main__":
    run_experiment()