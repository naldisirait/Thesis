import torch
import numpy as np
import torch.nn as nn

class BestModel:
    def __init__(self, best_model):
        """
        A class to store current best model
        """
        self.best_model = best_model

    def get_model(self):
        return self.best_model
    
    def set_model(self, new_best_model):
        self.best_model = new_best_model

torch.manual_seed(42)
def train_net(model,num_epochs,train_loader,valid_loader,optimizer, criterion, device):
    train_accuracy = []
    val_accuracy = []
    curr_acc = np.inf
    best_model = BestModel(model)

    for epoch in range(num_epochs):
        model.train()
        correct_train = 0  
        total_train = 0 
        for inputs, labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        t_acc = 100 * correct_train / total_train
        train_accuracy.append(t_acc)
        
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to("cpu")
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        v_acc = 100 * correct / total
        val_accuracy.append(v_acc)
        if v_acc < curr_acc:
            best_model.set_model(model)
            curr_acc = v_acc
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f},Train Acc: {t_acc:.2f},  Val Acc: {v_acc:.2f}%')

    return model,best_model.get_model(),train_accuracy,val_accuracy