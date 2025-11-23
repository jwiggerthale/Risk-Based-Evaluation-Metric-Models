from modules import heart_model
from data_utils import heart_ds
from examination_utils import get_tpr_fpr
from train_utils import predict
from get_scores import get_metrics

from torch.utils.data import DataLoader
import torch

import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline


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



# creat models with different temperatures for scaling
model_path = 'test_model.pth'
model_1 = heart_model(temperature = 1.0).double()
model_1.load_state_dict(torch.load(model_path))

model_5 = heart_model(temperature = 5.0).double()
model_5.load_state_dict(torch.load(model_path))

# get predictions for each temperature
pred_vals_1 = []
prob_vals_1 = []
label_vals_1 = []
for x, y in iter(train_loader):
    preds, probs = predict(model = model_1, 
                           x = x,
                          inference = True)
    pred_vals_1.extend(preds.flatten().tolist())
    prob_vals_1.extend(probs.flatten().tolist())
    label_vals_1.extend(y.tolist())


pred_vals_5 = []
prob_vals_5 = []
label_vals_5 = []
for x, y in iter(train_loader):
    preds, probs = predict(model = model_5, 
                           x = x,
                          inference = True)
    pred_vals_5.extend(preds.flatten().tolist())
    prob_vals_5.extend(probs.flatten().tolist())
    label_vals_5.extend(y.tolist())


# define parameters to plot wba
# global parameters for ROC Curve plot

P_to_N_ratios_powers = np.arange(-3, 4)
P_to_N_ratios_powers = np.array([-3, 0, 3])
P_to_N_ratios_powers = np.array([3])
P_to_N_ratios = 2.0 ** P_to_N_ratios_powers
number_of_ratios = len(P_to_N_ratios_powers)
plot_colors_roc = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.5]]
plot_colors_roc = [0.0, 0.0, 0.0]

linestyle = ':'

TPR_range = np.arange(0.0, 1.02, 0.02)
TNR_range = np.arange(0.0, 1.02, 0.02)
TPR, TNR = np.meshgrid(TPR_range, TNR_range)

face_alpha = 0.5
line_width = 4.0

# Create contours for grid
X = TPR
Y = TNR
line_width_grid = 2.0
grid_color = [0.2, 0.2, 0.2]
grid_levels = 4
ticks = np.arange(0.0, 1.2, 0.2)

# Plot options
use_wba = 1
use_f1score = 1
use_mcc = 1
use_fbscore = 1
dot_size = 12


numel = 51
num_el_factor = 1 / (numel - 1)
mid_el = (numel + 1) // 2
X_vals = np.linspace(0, 1, numel)

thresholds = np.arange(0, 1.02, 0.02)
tprs_1, fprs_1, tnrs_1, fnrs_1 = get_tpr_fpr(model = model_1, 
                      data_loader = val_loader, 
                      thresholds = thresholds, 
                                            inference = False)

tprs_5, fprs_5, tnrs_5, fnrs_5 = get_tpr_fpr(model = model_5, 
                      data_loader = val_loader, 
                      thresholds = thresholds, 
                                            inference = True)


wba_vals_5 = {}
P_to_N_ratios_powers = np.arange(-3, 4)
P_to_N_ratios_powers = np.array([-3, 0, 3])
P_to_N_ratios = 2.0 ** P_to_N_ratios_powers
#P_to_N_ratios = [1]
for c_fn in P_to_N_ratios:
    wba_vals_5[c_fn] =  []
    for i, fpr in enumerate(fprs_5):
        fpr_val, tpr_val, fnr_val, tnr_val, er_val, wba_val, precision_val, recall_val, f1_val = get_metrics(c_fn = c_fn, fpr_val = fpr, fnr_val = fnrs_5[i])
        wba_vals_5[c_fn].append(wba_val)


wba_vals_1 = {}
P_to_N_ratios_powers = np.arange(-3, 4)
P_to_N_ratios_powers = np.array([-3, 0, 3])
P_to_N_ratios = 2.0 ** P_to_N_ratios_powers
#P_to_N_ratios = [1]
for c_fn in P_to_N_ratios:
    wba_vals_1[c_fn] =  []
    for i, fpr in enumerate(fprs_1):
        fpr_val, tpr_val, fnr_val, tnr_val, er_val, wba_val, precision_val, recall_val, f1_val = get_metrics(c_fn = c_fn,  fpr_val = fpr, fnr_val = fnrs_1[i])
        wba_vals_1[c_fn].append(wba_val)


fig, axes = plt.subplots(1, 1, figsize = (15, 15))
for c_fn in P_to_N_ratios:
    plt.plot(thresholds, wba_vals_1[c_fn], label = str(c_fn))

plt.title('WBA Values Over Decsions Thresholds with Different Risk Ratios', fontsize = 24)
plt.xlabel('Threshold', fontsize = 18)
plt.ylabel('WBA', fontsize = 18)
plt.legend()
plt.savefig('./Risk_Based_Evaluation_Metric/ims/Test_WBAS_Thresholds_Uncalibrated.png')

fig, axes = plt.subplots(1, 1, figsize = (15, 15))
for c_fn in P_to_N_ratios:
    plt.plot(thresholds, wba_vals_5[c_fn], label = str(c_fn))

plt.title('WBA Values Over Decsions Thresholds with Different Risk Ratios', fontsize = 24)
plt.xlabel('Threshold', fontsize = 18)
plt.ylabel('WBA', fontsize = 18)
plt.legend()
plt.savefig('./Risk_Based_Evaluation_Metric/ims/Test_WBAS_Thresholds_Calibrated.png')


# get stats for both models
wba_vals_5 = {}
er_vals_5 = {}
P_to_N_ratios_powers = np.arange(-4, 5)
P_to_N_ratios = 2.0 ** P_to_N_ratios_powers
for c_fn in P_to_N_ratios:
    wba_vals_5[c_fn] =  []
    er_vals_5[c_fn] =  []
    for i, fpr in enumerate(fprs_5):
        fpr_val, tpr_val, fnr_val, tnr_val, er_val, wba_val, precision_val, recall_val, f1_val = get_metrics(c_fn = c_fn, fpr_val = fpr, fnr_val = fnrs_5[i])
        wba_vals_5[c_fn].append(wba_val)
        er_vals_5[c_fn].append(er_val)


wba_vals_1 = {}
er_vals_1 = {}
P_to_N_ratios_powers = np.arange(-4, 5)
P_to_N_ratios = 2.0 ** P_to_N_ratios_powers
for c_fn in P_to_N_ratios:
    wba_vals_1[c_fn] =  []
    er_vals_1[c_fn] =  []
    for i, fpr in enumerate(fprs_1):
        fpr_val, tpr_val, fnr_val, tnr_val, er_val, wba_val, precision_val, recall_val, f1_val = get_metrics(c_fn = c_fn, fpr_val = fpr, fnr_val = fnrs_1[i])
        wba_vals_1[c_fn].append(wba_val)
        er_vals_1[c_fn].append(er_val)


stats = {}
for c_fn in P_to_N_ratios:
    max_wba = np.max(wba_vals_1[c_fn])
    max_wba_s = np.argmax(wba_vals_1[c_fn])
    max_wba_s = thresholds[max_wba_s]
    default_wba = wba_vals_1[c_fn][50]
    min_er = np.min(er_vals_1[c_fn])
    min_er_s = np.argmin(er_vals_1[c_fn])
    min_er_s = thresholds[min_er_s]
    default_er = er_vals_1[c_fn][50]
    stats[c_fn] = [[max_wba, max_wba_s, default_wba], [min_er, min_er_s, default_er]]

with open('stats_uncalibrated_model.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(stats, out_file)

stats = {}
for c_fn in P_to_N_ratios:
    max_wba = np.max(wba_vals_5[c_fn])
    max_wba_s = np.argmax(wba_vals_5[c_fn])
    max_wba_s = thresholds[max_wba_s]
    default_wba = wba_vals_5[c_fn][50]
    min_er = np.min(er_vals_5[c_fn])
    min_er_s = np.argmin(er_vals_5[c_fn])
    min_er_s = thresholds[min_er_s]
    default_er = er_vals_5[c_fn][50]
    stats[c_fn] = [min_er, min_er_s, default_er, default_er/min_er, max_wba, default_wba]

with open('stats_calibrated_model.json', 'w', encoding = 'utf-8') as out_file:
    json.dump(stats, out_file)