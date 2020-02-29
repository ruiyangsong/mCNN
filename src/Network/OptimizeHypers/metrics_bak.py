#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
!! ATTENTION !!
*For those custom metrics, the average accross minibatches is namely not equal to the metric evaluated on the whole dataset.
*The metric on the validation set is calculated in batches, and then averaged (of course the trained model at the end of the epoch is used,
 in contrast to how the metric score is calculated for the training set)
1. How to compute precision and recall in Keras? --> https://www.thinbug.com/q/43076609
2. How are metrics computed in Keras? --> https://stackoverflow.com/questions/49359489/how-are-metrics-computed-in-keras
'''
from keras import backend as K

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def acc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tp + tn) / (tp + tn + fp + fn + K.epsilon())

def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print(tp,fp,tn,fn)
    return numerator / (denominator + K.epsilon())


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def recall_p(y_true,y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return tp/(tp + fn + K.epsilon())

def recall_n(y_true,y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    return tn / (tn + fp + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def precision_p(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    return tp/(tp + fp + K.epsilon())

def precision_n(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg)
    fn = K.sum(y_pos * y_pred_neg)
    return tn/(tn + fn + K.epsilon())

def test_report(model,x_test,y_test):
    import numpy as np
    p_pred = model.predict(x_test, batch_size=32, verbose=0)  # 测试数据属于每一个类的概率,ndarray
    y_pred = np.argmax(p_pred, axis=1)  # 0D array
    y_real = np.argmax(y_test, axis=1)
    # print(y_real.shape) # 1D
    # print(y_pred.shape) # 1D
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(y_pred.shape[0]):
        if y_real[i] == 1 and y_pred[i] == 1:
            tp+=1
        elif y_real[i] == 1 and y_pred[i] == 0:
            fn+=1
        elif y_real[i] == 0 and y_pred[i] == 0:
            tn+=1
        elif y_real[i] == 0 and y_pred[i] == 1:
            fp+=1
    acc = (tp+tn)/(tp+tn+fp+fn+K.epsilon())
    recall_p = tp/(tp+fn+K.epsilon())
    recall_n = tn/(tn+fp+K.epsilon())
    precision_p = tp/(tp+fp+K.epsilon())
    precision_n = tn/(tn+fn+K.epsilon())
    mcc = (tp*tn-fp*fn)/(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))+K.epsilon())
    return acc, mcc, recall_p,recall_n,precision_p,precision_n

def tp_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp

def tn_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tn

def fp_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fp

def fn_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fn

def acc_Concise(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn) / (tp + tn + fp + fn)

def recall_p_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp/(tp + fn + K.epsilon())

def recall_n_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp + K.epsilon())

def precision_p_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp/(tp + fp + K.epsilon())

def precision_n_Concise(y,z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tn/(tn + fn + K.epsilon())

def mcc_concise(y,z):
    """Matthews correlation coefficient
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) / K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def contingency_table(y, z):
    """Note:  if y and z are not rounded to 0 or 1, they are ignored
    """
    y = K.cast(K.round(y), K.floatx())
    z = K.cast(K.round(z), K.floatx())

    def count_matches(y, z):
        return K.sum(K.cast(y, K.floatx()) * K.cast(z, K.floatx()))

    ones = K.ones_like(y)
    zeros = K.zeros_like(y)
    y_ones = K.equal(y, ones)
    y_zeros = K.equal(y, zeros)
    z_ones = K.equal(z, ones)
    z_zeros = K.equal(z, zeros)

    tp = count_matches(y_ones, z_ones)
    tn = count_matches(y_zeros, z_zeros)
    fp = count_matches(y_zeros, z_ones)
    fn = count_matches(y_ones, z_zeros)

    return (tp, tn, fp, fn)