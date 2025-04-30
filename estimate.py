def calculate_confusion_matrix(y_hat, y):
    y_hat = np.array(y_hat)
    y = np.array(y)

    n, m = y_hat.shape
    TP = np.zeros(m)
    FP = np.zeros(m)
    TN = np.zeros(m)
    FN = np.zeros(m)

    for h in range(m):
        for v in range(n):
            if y_hat[v, h] == 1 and y[v, h] == 1:
                TP[h] += 1
            elif y_hat[v, h] == 1 and y[v, h] == 0:
                FP[h] += 1
            elif y_hat[v, h] == 0 and y[v, h] == 1:
                FN[h] += 1
            elif y_hat[v, h] == 0 and y[v, h] == 0:
                TN[h] += 1

    return TP, FP, TN, FN


def Aiming(y_hat, y):
    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:  # L ∪ L*
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:  # L ∩ L*
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y_hat[v])
    return score_k / n


def Coverage(y_hat, y):
    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y[v])

    return score_k / n


def Accuracy(y_hat, y):
    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / union
    return score_k / n


def AbsoluteFalse(y_hat, y):
    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        score_k += (union - intersection) / m
    return score_k / n


import numpy as np

def evaluate(score_label, y, threshold=None):
    y_hat = score_label.copy()
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    # Calculate confusion matrix
    TP, FP, TN, FN = calculate_confusion_matrix(y_hat, y)
    return aiming, coverage, accuracy, absolute_false, TP, FP, TN, FN





