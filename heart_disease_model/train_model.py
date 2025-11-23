import pandas as pd
import numpy as np
import copy

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline


from modules import heart_model
from data_utils import heart_ds
from train_utils import train_loop




# get data and define column types
data = pd.read_csv('./heart_disease/heart.csv')
target_col = 'HeartDisease'
numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categoric_cols = [c for c in data.columns.values.tolist() if c not in numeric_cols and c != target_col]

feature_cols = copy.deepcopy(numeric_cols)
feature_cols.extend(categoric_cols)

# create training and test data 
idx = data.index.tolist()
np.random.shuffle(idx)

# for deterministic behavior save indexes
# --> always use identical data for training, testing and validation
# just load index for future tests 
for i in idx :
    with open('index_file.txt', 'a', encoding = 'utf-8') as out_file:
        out_file.write(f'{i}\n')

with open('index_file.txt', 'r', encoding = 'utf-8') as in_file:
    idx = in_file.readlines()
idx = [int(i) for i in idx]
    
num_train_samples = int(0.8 * len(data))
num_test_samples = int(0.9 * len(data))
idx_train  = idx[:num_train_samples]
idx_val = idx[num_train_samples:num_test_samples]
idx_test = idx[num_test_samples:]


x_test = data.loc[idx_test, feature_cols]
x_val = data.loc[idx_val, feature_cols]
x_train = data.loc[idx_train, feature_cols]

y_test = data.loc[idx_test, target_col]
y_val = data.loc[idx_val, target_col]
y_train = data.loc[idx_train, target_col]


num_pipe = Pipeline([
    ("scaler",  RobustScaler())
])
cat_pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num",  num_pipe,  numeric_cols),
        ("cat",  cat_pipe,  categoric_cols),
    ],
    remainder="drop"
)


x_train = preprocess.fit_transform(x_train)
x_val   = preprocess.transform(x_val)
x_test  = preprocess.transform(x_test)

# create train, val and test set and wrap into a data loader
train_set = heart_ds(x = x_train, y = y_train.tolist())
val_set = heart_ds(x = x_val, y = y_val.tolist())
test_set = heart_ds(x = x_test, y = y_test.tolist())

train_loader = DataLoader(train_set, batch_size = 32)
test_loader = DataLoader(test_set, batch_size = 32)
val_loader = DataLoader(val_set, batch_size = 32)



# creat model and train
# model is automatically saved to model_path
model_path = 'test_model.pth'
model = heart_model(temperature = 1.0).double()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
train_loop(train_loader = train_loader, 
           val_loader = val_loader,  
           model = model, 
           criterion = criterion, 
           optimizer = optimizer,
           num_epochs = 100, 
           early_stopping = 20, 
           log_file = 'log.txt', 
           device = 'cpu', 
           model_name = model_path)