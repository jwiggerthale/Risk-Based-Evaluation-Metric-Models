import math


def er(c_fn: float, 
      fpr_val: float, 
      fnr_val: float):
    er = fpr_val + c_fn * fnr_val
    return er

def wba(c_fn: float, 
      fpr_val: float, 
      fnr_val: float):
    wba = ((1 - fpr_val) + c_fn * (1 - fnr_val))/ (1 + c_fn)
    return wba

def precision(tpr_val: float, 
             fpr_val: float):
    precision = tpr_val / (tpr_val + fpr_val)
    return precision

def recall(tpr_val: float, 
          fnr_val: float):
    recall  = tpr_val / (tpr_val + fnr_val)
    return recall

def f1_score(precision_val: float, 
             recall_val: float):
    f1 = 2 * precision_val * recall_val / (precision_val + recall_val)
    return f1


def get_metrics(c_fn: float, 
                fpr_val: float, 
                fnr_val: float):
    tnr_val = 1 - fpr_val
    tpr_val = 1 - fnr_val
    er_val = er(c_fn = c_fn, 
                fpr_val = fpr_val, 
                fnr_val = fnr_val)
    wba_val = wba(c_fn = c_fn, 
                fpr_val = fpr_val, 
                fnr_val = fnr_val)
    precision_val = precision(tpr_val = tpr_val, 
                              fpr_val = fpr_val)
    recall_val = recall(tpr_val = tpr_val, 
                    fnr_val = fnr_val)
    f1_val = f1_score(precision_val = precision_val, 
                      recall_val = recall_val)
    return fpr_val, tpr_val, fnr_val, tnr_val, er_val, wba_val, precision_val, recall_val, f1_val
    