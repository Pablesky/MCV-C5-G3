# https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k

import numpy as np
from sklearn.metrics import average_precision_score

def precission_k(true_class, class_preds):
    true_class = np.array(true_class)
    class_preds = np.array(class_preds)

    # Get the number of class preds equal to the true class
    true_preds = np.sum((class_preds == true_class).astype(int))

    return true_preds / len(class_preds)

def recall_k(true_class, class_preds, number_true):
    true_class = np.array(true_class)
    class_preds = np.array(class_preds)

    # Get the number of class preds equal to the true class
    true_preds = np.sum((class_preds == true_class).astype(int))

    return true_preds / number_true

def ap(true_class, class_preds):
    true_class = np.array(true_class)
    class_preds = np.array(class_preds)

    return average_precision_score(true_class, class_preds)


if __name__ == '__main__':
    true_class = [1]
    pred_class = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    print(recall_k(true_class, pred_class[:5], 8))
