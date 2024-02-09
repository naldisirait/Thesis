import torch
import torch.nn as nn
import numpy as np
from .evaluation import evaluate_model_simvp

def train_SimVP(config_class, epochs, criterion, optimizer, model, data_train_loader, data_val_loader, device):
    exp_name = config_class.get_experiment_name()
    prev_epochs = 500
    model.to(device)
    model.train()
    prev_loss = np.inf
    eval_loss = []
    train_loss = []
    for epoch in range(epochs):
        for X, y in data_train_loader:
            X = X.to(device)
            y = y.to(device)
            #forward
            output = model(X)
            #Calculate MSE
            loss = criterion(output,y)
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #Check error using data validation      
        mse_eval, predictions, real = evaluate_model_simvp(model, data_val_loader)
        eval_loss.append(mse_eval)
        train_loss.append(loss.item())
        
        print(f"Epoch {epoch + 1}/{epochs} with loss train {loss.item()}, validation {mse_eval}")
        
        if mse_eval < prev_loss:
            prev_loss = mse_eval
            #Save model
            torch.save(model.state_dict(),f"{path_to_save} SimVP exp {exp_name} epoch {epochs+prev_epochs} best.pt")
            
    torch.save(model.state_dict(),f"SimVP run 3 epoch {epochs+prev_epochs}.pt") 