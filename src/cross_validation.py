#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : cross_validation.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from processing import *
from train_model import train_model
from test_model import test_model

def cross_validation(x, y, ddg, k, random_seed, nn_model, normalize_method, train_ratio=0.7):
    '''
    :param x: 3D numpy array, stored numerical representation of this dataset.
    :param y: 1D numpy array, labels of x.
    :param ddg: 0D numpy array, ddg array.
    :param k: int, fold number.
           k == 0 --> when giving blind data manually.
           k == 1 --> blind test.
           k == 2 --> when giving k_fold data manually.
           k >= 3 --> kfold cross validation.
    :param random_seed: int, random seed for shuffle k_fold and split_val.
    :param nn_model: float, which network structure to choose.
           nn_model == 1.xx --> classification task.
           nn_model == 2.xx --> regression task.
    :param normalize_method: str, 'max' or 'norm'.
    :param train_ratio: float, split ratio for blind test. 0.7 is the default option.
    :return: label balanced train data and test data.
    '''
    k_seed, v_seed = random_seed
    kfold_score = np.zeros((k, 6))
    k_count = 1
    history_list = []

    ## giving blind data manually.
    if k == 0:
        pass

    ## blind test.
    if k == 1:
        positive_indices, negative_indices = ddg >= 0, ddg < 0
        x_positive, x_negative = x[positive_indices], x[negative_indices]
        y_positive, y_negative = y[positive_indices], y[negative_indices]
        ddg_positive, ddg_negative = ddg[positive_indices], ddg[negative_indices]
        left_positive, left_negative = round(0.7 * x_positive.shape[0]), round(0.7 * x_negative.shape[0])
        x_train, x_test = np.vstack((x_positive[:left_positive], x_negative[:left_negative])), np.vstack(
            (x_positive[left_positive:], x_negative[left_negative:]))
        y_train, y_test = np.vstack((y_positive[:left_positive], y_negative[:left_negative])), np.vstack(
            (y_positive[left_positive:], y_negative[left_negative:]))
        ddg_train, ddg_test = np.vstack((ddg_positive[:left_positive], ddg_negative[:left_negative])), np.vstack(
            (ddg_positive[left_positive:], ddg_negative[left_negative:]))
        x_train, y_train, ddg_train = shuffle_data(x_train, y_train, ddg_train, v_seed)
        x_test, y_test, ddg_test = shuffle_data(x_test, y_test, ddg_test, v_seed)
        ## train model.

    ## giving k_fold data manually.
    if k == 2:
        pass

    ## k_fold cross validation.
    if k >= 3:
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = k_seed)
        for train_index, test_index in skf.split(x, y):
            print('%d is in progress, total %d' % (k_count, k))
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ddg_train, ddg_test = ddg[train_index], ddg[test_index]
            ## train model on each fold.
            network, history_dict, x_test, y_test, ddg_test = train_model(
                x_train, y_train, ddg_train, x_test, y_test, ddg_test, v_seed, nn_model, normalize_method)

            history_list.append(history_dict)

            ## test model on each fold.
            if nn_model < 2:
                acc, recall_p, recall_n, precision_p, precision_n, mcc = test_model(
                    network, x_test, y_test, ddg_test, nn_model)
                kfold_score[k_count - 1, :] = [acc, recall_p, recall_n, precision_p, precision_n, mcc]
            elif nn_model > 2:
                pearson_coeff, rmse = test_model(network, x_test, y_test, ddg_test, nn_model)
                kfold_score[k_count - 1, :2] = [pearson_coeff, rmse]
            k_count += 1

    return kfold_score, history_list

if __name__ == '__main__':
    ## Input parameters.
    dataset_name, radius, k_neighbor, class_num, k, nn_model, normalize_method, sort_method,\
    p_seed, k_seed, v_seed = sys.argv[1:]

    radius = float(radius)
    k_neighbor = int(k_neighbor)
    class_num = int(class_num) # class number of atoms.
    k = int(k) # kfold
    nn_model = float(nn_model) # which CNN structure to choose.
    seed = [int(p_seed), int(k_seed), int(v_seed)] # seeds for permutation, split k_fold, split val.

    ## load data
    x, y, ddg = load_data(dataset_name,radius,k_neighbor,class_num)
    print('Loading data from hard drive is done.')

    ## sort row of each mutation matrix.
    x = sort_row(x, sort_method, seed[0])
    print('Sort row is done, sorting method is %s.' % sort_method)

    ## Cross validation.
    print('Cross validation begin, total k is %d' % k)
    kfold_score, history_list = cross_validation(x, y, ddg, k, seed[1:], nn_model, normalize_method, train_ratio=0.7)

    print_result(nn_model,kfold_score)
    ## plot.
    #plotfigure(history_dict)