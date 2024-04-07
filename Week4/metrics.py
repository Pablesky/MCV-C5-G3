# https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k

import numpy as np
from sklearn.metrics import average_precision_score


def compute_metrics(true_class, class_preds, number_true):

    positive_accumulation = 0
    ap = 0
    rec = []
    prec = []

    for i in range(len(class_preds)):

        if true_class == class_preds[i]:
            positive_accumulation += 1
            ap += positive_accumulation / (i + 1)
        
        prec.append(positive_accumulation / (i + 1))
        rec.append(positive_accumulation / number_true)
    
    ap = ap / number_true

    return ap, prec, rec