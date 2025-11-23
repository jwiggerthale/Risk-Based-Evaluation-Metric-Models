'''
This script trains a model on the chest x-ray dataset from kaggle and visualizes model performacne on test data in a confusion matrix
You can specify model type to be trained by adapting variable model_name
'''

import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim

from data_utils import transform_test, transform_train, label_to_num, lung_dataset
from test_model import test_model
from modules import VGG16, resnet18, resnext50
from train_utils import train_loop

train_dir = './train'
test_dir = './test'
val_dir = './val'

train_files = []
val_files = []
test_files = []
for class_name in ['normal', 'pneumonia', 'tuberculosis']:
    use_dir = f'{train_dir}/{class_name}'
    new_files = [f'{use_dir}/{f}' for f in os.listdir(use_dir) if f.endswith('jpg')]
    train_files.extend(new_files)
    use_dir = f'{test_dir}/{class_name}'
    new_files = [f'{use_dir}/{f}' for f in os.listdir(use_dir) if f.endswith('jpg')]
    test_files.extend(new_files)
    use_dir = f'{val_dir}/{class_name}'
    new_files = [f'{use_dir}/{f}' for f in os.listdir(use_dir) if f.endswith('jpg')]
    val_files.extend(new_files)




train_set = lung_dataset(im_files = train_files, 
                         transform = transform_train)

test_set = lung_dataset(im_files = test_files, 
                         transform = transform_test)

val_set = lung_dataset(im_files = val_files, 
                         transform = transform_test)

train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)


model_name = 'resnet_3_classes'
if 'vgg' in model_name.lower():
    model = VGG16(num_classes=3)
    model.freeze_weights()
elif 'resnet' in model_name.lower():
    model = resnet18(num_classes=3)
    model.freeze_weights(layers = [1,2,3])
elif 'resnext' in model_name.lower():
    model = resnext50(num_classes=3)
    model.freeze_weights(layers = [1,2,3])


model = model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

train_loop(train_loader = train_loader, 
           val_loader = val_loader, 
           model = model, 
           criterion = criterion, 
           optimizer = optimizer, 
           save_name=model_name, 
           log_file=f'log_{model_name}.txt', 
           num_epochs=50, 
           early_stopping=5
)

model.load_state_dict(torch.load(f'{model_name}_Acc.pth'))


test_model(model = model, 
           data_loader = test_loader, 
           save_name = f'{model_name}_Acc_test.png', 
           criterion = criterion)

model.load_state_dict(torch.load(f'{model_name}_Loss.pth'))



test_model(model = model, 
           data_loader = test_loader, 
           save_name = f'{model_name}_Loss.png', 
           criterion = criterion)


