'''
This script implements a function to test the model on a given dataset
Model makes prediction on all instances and confusion matrix is saved
'''

import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

from data_utils import transform_test, transform_train, label_to_num, lung_dataset
from modules import VGG16, resnet18
from train_utils import val_step

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


def test_model(model: nn.Module, 
               data_loader: DataLoader, 
               save_name: str = 'cm_vgg.png', 
               model_name: str = 'VGG',
               criterion: nn.Module = nn.CrossEntropyLoss()):
    val_acc = 0.0
    val_loss = 0.0
    preds = []
    labels = []
    for x, y in iter(data_loader):
        labels.extend(y.tolist())
        loss, acc, _, pred = val_step(x = x, 
                                y = y, 
                                model = model, 
                                criterion = criterion, 
                                device = 'cuda')
        val_loss += loss
        val_acc += acc
        preds.extend(pred.tolist())

    val_loss /= len(data_loader)
    val_acc /= len(data_loader)
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0.5, 1.5, 2.5], list(label_to_num.keys()))
    plt.yticks([0.5, 1.5, 2.5], list(label_to_num.keys()))
    plt.title(f'Confusion Matrix of {model_name} Model')
    plt.savefig(save_name)
    plt.close()
    disp = ConfusionMatrixDisplay(cm, display_labels=list(label_to_num.keys()))
