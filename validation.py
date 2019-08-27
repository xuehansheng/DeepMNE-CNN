# !/usr/bin/env python
# -*- coding: utf8 -*-
# author:xuehansheng(xhs1892@gmail.com)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy_top(scores, labels):
    predictions = scores.argmax(axis=1)
    count_ = 0
    for i in range(len(predictions)):
        if labels[i, predictions[i]] == 1.0:
            count_ = count_ + 1
    acc = count_ / float(len(predictions))
    return acc

def real_AUPR(label, score):
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0

    return pr, f

def evaluate_performance(y_test, y_score, y_pred):
    n_classes = y_test.shape[1]
    perf = dict()
    # Compute macro-averaged AUPR
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    perf["M-aupr"] /= n

    # Compute micro-averaged AUPR
    perf["m-aupr"], _ = real_AUPR(y_test, y_score)

    # Computes accuracy
    # perf['acc'] = accuracy_score(y_test, y_pred)
    perf['acc'] = accuracy_top(y_score, y_test)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    # print y_new_pred.shape
    for i in range(y_pred.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        # for idx in range(len(top_alpha)):
        #     y_new_pred[i, top_alpha[idx]] = np.array(alpha*[1])
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1"] = f1_score(y_test, y_new_pred, average='micro')

    return perf
