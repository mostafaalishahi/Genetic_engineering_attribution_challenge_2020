import numpy as np
from sklearn import metrics

def top10_accuracy_scorer(y_true, y_probs, target_size):
    y_true = np.array(y_true)
    top10_idx = np.argpartition(y_probs, -10, axis=1)[:, -10:]
    top10_preds = np.array([i for i in range(target_size)])[top10_idx]
    mask = top10_preds == y_true.reshape((y_true.size, 1))
    top_10_accuracy = mask.any(axis=1).mean()
    return top_10_accuracy

def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred, average='macro'):
    return metrics.f1_score(y_true, y_pred, average=average)

def print_all_metrics(y_true, y_probs, target_size, average='macro'):
    y_pred = np.argmax(y_probs, axis=1)
    acc = accuracy(y_true, y_pred)
    top10 = top10_accuracy_scorer(y_true, y_probs, target_size=target_size)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Acc: {:.4f}, top10: {:.4f}, F1: {:.4f}'.format(acc, top10, f1))
    return True