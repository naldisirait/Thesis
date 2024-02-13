import torch
import torch.nn as nn
import numpy as np
from .evaluation import evaluate_model_simvp

def train_SimVP(config_class, model, data_train_loader, data_val_loader, device):
    
    epochs = config_class.get_epochs()
    criterion = config_class.get_criterion()
    optimizer = config_class.get_optimzer()

    model.to(device)
    model.train()

    #set best_model to 0 
    best_model = 0
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
            best_model = model

    return model,best_model, train_loss, eval_loss