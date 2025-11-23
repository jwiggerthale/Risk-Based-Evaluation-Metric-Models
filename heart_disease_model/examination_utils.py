import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def get_tpr_fpr(model: nn.Module,  
                data_loader: DataLoader,
                thresholds: np.array, 
               inference: bool = True):
    pred_vals = []
    labels = []
    for x, y in iter(data_loader):
        pred = model.forward(x, inference=inference)
        pred_vals.extend(pred.tolist())
        labels.extend(y.tolist())
    
    tprs = []
    fprs = []
    tnrs = []
    fnrs = []
    
    for s in thresholds:
        cm = confusion_matrix(np.array(labels), 
                             (np.array(pred_vals) >= s).astype(int).reshape(-1))
        
        # Confusion Matrix: cm[true_label, predicted_label]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
        
        # Berechne Metriken mit den richtigen Nennern
        # TPR und FNR: normiert auf alle tatsächlich Positiven
        total_positives = TP + FN
        tpr = TP / total_positives if total_positives > 0 else 0.0
        fnr = FN / total_positives if total_positives > 0 else 0.0
        
        # FPR und TNR: normiert auf alle tatsächlich Negativen
        total_negatives = FP + TN
        fpr = FP / total_negatives if total_negatives > 0 else 0.0
        tnr = TN / total_negatives if total_negatives > 0 else 0.0
        
        tprs.append(tpr)
        fprs.append(fpr)
        tnrs.append(tnr)
        fnrs.append(fnr)
    
    return tprs, fprs, tnrs, fnrs