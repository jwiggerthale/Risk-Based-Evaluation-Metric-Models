from utils import log
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def train_step(x: torch.tensor,
               y: torch.tensor, 
               device: str,
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module):
    model.train()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits = model(x).flatten()
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()



'''
function to make prediction with model
call with: 
    x: torch.tensor --> image you want to predict on
    model: nn.Module --> model to be applied
returns: 
    preds: torch.tensor --> predicted classes
    logits: torch.tensor --> raw model output
'''
def predict(x: torch.tensor, 
            model: nn.Module, 
           inference: bool = False):
    model.eval()
    with torch.no_grad():
        logits = model(x, inference = inference)
        preds = logits.round()
    return preds, logits



def val_step(x: torch.tensor, 
             y: torch.tensor, 
             model: nn.Module,
             criterion: nn.Module, 
             device: str = 'cuda'):
    x, y = x.to(device), y.to(device)
    preds, probs = predict(model = model, 
                           x = x)
    preds = preds.flatten()
    probs = probs.flatten()
    loss = criterion(probs, y)
    acc = (preds == y).sum()/len(x)
    wrong = len(x) - (preds == y).sum()
    return loss, acc, wrong



def train_loop(train_loader: DataLoader, 
               val_loader: DataLoader, 
               model: nn.Module, 
               criterion: nn.Module, 
               optimizer: nn.Module,
               aug_loader: DataLoader = None,
               num_epochs: int = 100, 
               early_stopping: int = 20, 
               log_file: str = 'log.txt', 
               device: str = 'cuda', 
               model_name: str = 'heart_model.pth'):
    best_acc = 0.0
    best_loss = np.inf
    counter = 0
    for epoch in range(num_epochs):
        counter += 1
        running_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        for x, y in iter(train_loader):
            loss = train_step(x = x, 
                              y = y, 
                              model = model, 
                              criterion = criterion, 
                              optimizer = optimizer, 
                              device = device)
            running_loss += loss
        running_loss /= len(train_loader)
        if aug_loader is not None:
            running_loss *= len(train_loader)
            for x, y in iter(aug_loader):
                loss = train_step(x = x, 
                                y = y, 
                                model = model, 
                                criterion = criterion, 
                                optimizer = optimizer, 
                                device = device)
                running_loss += loss
            running_loss /= (len(train_loader) + len(aug_loader))
        log(f'training in epoch {epoch +1 } completed: ; loss: {running_loss}', 
            file = log_file)
        for x, y in iter(val_loader):
            loss, acc, _ = val_step(x = x, 
                                    y = y, 
                                    model = model, 
                                    criterion = criterion, 
                                    device = device)
            val_loss += loss
            val_acc += acc
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        log(f'validation in epoch {epoch +1 } completed: acc: {val_acc}; loss: {val_loss}', 
            file=log_file)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'new best model in epoch {epoch + 1}: accuracy: {val_acc}')
            counter = 0
            torch.save(model.state_dict(), model_name)
        elif val_loss < best_loss:
            best_loss = val_loss
            print(f'new best model in epoch {epoch +1 }: val loss: {val_loss}')
            counter = 0
            torch.save(model.state_dict(), model_name)
        elif(counter > early_stopping):
            #print(f'No improvement in {early_stopping} epochs; Training interrupted in epoch {epoch+1}')
            break


   


    