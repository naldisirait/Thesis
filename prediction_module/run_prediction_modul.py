from source.ingest_data import load_dataset
from source.Configuration import Configuration
from source.data_processing import process_dataset
from source.building_data_loader import create_data_loader
from source.model import create_model_prediction
from source.train_model import train_model
from utils.utils import read_json_file
from utils.utils import load_pretrained_model
import torch

def run_experiment():
    #set device
    device = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")

    #load configuration
    config = read_json_file("config.json")
    config_class = Configuration(config)

    #Load dataset
    path_dataset = config_class.get_path_dataset()
    list_of_train_dataset = config_class.get_list_of_train_data()
    list_of_validation_dataset = config_class.get_list_of_validation_data()
    train_dataset = load_dataset(path = path_dataset, list_of_filename=list_of_train_dataset)
    val_dataset = load_dataset(path = path_dataset, list_of_filename=list_of_validation_dataset)

    #process dataset
    X_train, y_train = process_dataset(train_dataset)
    X_val, y_val = process_dataset(val_dataset)

    #create data loader
    batch_size = config_class.get_batch_size()
    data_train_loader = create_data_loader(X=X_train, y=y_train, batch_size=batch_size, shuffle=True)
    data_val_loader = create_data_loader(X=X_val, y=y_val, batch_size=batch_size, shuffle=False)
    
    #create model
    model = create_model_prediction(config_class=config_class, in_shape=X_train.shape)

    #model = load_pretrained_model(model)
    model, best_model, train_loss, eval_loss = train_model(config_class, model, data_train_loader, data_val_loader)

    print("Successfully run the experiment!")

if __name__ == "__main__":
    run_experiment()