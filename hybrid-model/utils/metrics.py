import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def calculate_accuracy(predictions, batch_y):
    return np.mean(predictions == batch_y)


def calculate_confusion_matrix_values(preds, trues):
    tp = np.sum(np.logical_and(preds == 1.0, trues == 1.0))
    fp = np.sum(np.logical_and(preds == 1.0, trues == 0.0))
    tn = np.sum(np.logical_and(preds == 0.0, trues == 0.0))
    fn = np.sum(np.logical_and(preds == 0.0, trues == 1.0))

    return tp, fp, tn, fn


def calculate_precision(preds, trues):
    tp, fp, tn, fn = calculate_confusion_matrix_values(preds, trues)

    precision = tp / (tp + fp)
    precision = np.nan_to_num(precision)
    return precision


def calculate_recall(preds, trues):
    tp, fp, tn, fn = calculate_confusion_matrix_values(preds, trues)

    recall = tp / (tp + fn)
    recall = np.nan_to_num(recall)
    return recall


def calculate_f1(preds, trues):
    precision = calculate_precision(preds, trues)
    recall = calculate_recall(preds, trues)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    return f1


def calculate_specificity(preds, trues):
    tp, fp, tn, fn = calculate_confusion_matrix_values(preds, trues)

    specificity = tn / (tn + fp)
    specificity = np.nan_to_num(specificity)
    return specificity
