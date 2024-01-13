from source.data_ingest import load_dataset
from source.data_processing import data_processing
from source.building_data_loader import create_data_loader
from source.model_net import create_model_efficient_net
from source.train_model import train_net
import torch

def run_experiment():
    #get the index
    gpu_idx = 1
    device = torch.device(f"cuda:{gpu_idx}") if torch.cuda.is_available() else torch.device("cpu")

    path_dataset = ".."
    file_train = [".."]
    file_val = [".."]

    #load dataset
    train_dataset = load_dataset(path_dataset = path_dataset, data_name=file_train)
    val_dataset = load_dataset(path_dataset=path_dataset, data_name= file_val)

    #processing dataset
    X_train, y_train, X_val, y_val = data_processing(train_dataset=train_dataset,val_dataset=val_dataset)

    #Create custom data loader
    batch_size = 64
    train_loader = create_data_loader(X = X_train, y = y_train, batch_size = batch_size, shuffle=True)
    val_loader = create_data_loader(X = X_val, y = y_val, batch_size = batch_size, shuffle = False)

    #create the model
    version = "b0"
    num_classes = 6
    model = create_model_efficient_net(version=version, num_classes=num_classes)
    
    #train the model
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr)

    model,train_accuracy,val_accuracy = train_net(model, 
                                                  num_epochs=num_epochs,
                                                  train_loader= train_loader,
                                                  valid_loader= val_loader,
                                                  criterion= criterion,
                                                  optimizer= optimizer,
                                                  device=device)

if __name__ == "__main__":
    run_experiment()