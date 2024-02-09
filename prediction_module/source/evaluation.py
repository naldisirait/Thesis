import torch
import torch.nn as nn
import numpy as np

def evaluate_model_simvp(model, data_test_loader, device, order_seq = "all"):
    """
    Evaluate a PyTorch model using Mean Squared Error (MSE).

    Args:
        model : The neural network model to evaluate.
        data_test_loader : DataLoader containing the evaluation dataset.

    Returns:
        float: The Mean Squared Error (MSE) of the model on the evaluation dataset.
    """
    
    model.eval()  # Set the model to evaluation mode (turn off dropout, etc.)
    mse_loss = nn.MSELoss()  # Create the MSE loss function
    
    total_loss = []
    predicted = []
    real = []
    
    with torch.no_grad():  # Disable gradient tracking during evaluation
        for i, (inputs, targets) in enumerate(data_test_loader):
            inputs, targets = inputs.to(device), targets.to(device) # Ensure inputs and targets are on the same device with model
            outputs = model(inputs)  # Forward pass
            
            # Compute the MSE loss
            if order_seq == "all":
                loss = mse_loss(outputs, targets)

                # Update total loss and the number of samples
                total_loss.append(loss.item())
            else:
                n = order_seq
                loss = mse_loss(outputs[:,n,:,:,:], targets[:,n,:,:,:])
                total_loss.append(loss.item())
            targets = targets.to("cpu")
            outputs = outputs.to("cpu")
            if i == 0:
                real = targets
                predicted = outputs
            else:
                real = torch.cat((real,targets), dim = 0)
                predicted = torch.cat((predicted,outputs), dim = 0)
                
    # Calculate the average MSE
    average_mse = sum(total_loss) / len(total_loss)
    
    model.train()
    return average_mse, predicted, real