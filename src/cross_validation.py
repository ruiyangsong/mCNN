#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : cross_validation.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import numpy as np
from sklearn.model_selection import StratifiedKFold
from processing import shuffle_data

def cross_validation(x, y, ddg, k, seed, nn_model, train_ratio=0.7):
    '''
    :param x: 3D numpy array, stored numerical representation of this dataset.
    :param y: 1D numpy array, labels of x.
    :param ddg: 0D numpy array, ddg array.
    :param k: int, fold number.
           k == 0 --> when giving the k_fold data manually.
           k == 2 --> blind test.
           k >= 3 --> kfold cross validation.
    :param seed: int, random seed for shuffle x.
    :param nn_model: float, which network structure to choose.
           nn_model == 1.xx --> classification task.
           nn_model == 2.xx --> regression task.
    :param train_ratio: float, split ratio for blind test. 0.7 is the default option.
           train_ratio == 0 --> when giving blind data manually.
           train_ratio > 0  --> split train and test data for blind test.
    :return: label balanced train data and test data.
    '''
    ## giving k_fold data manually.
    if k == 0:
        pass
    if k == 2 and train_ratio == 0:
        pass
    ## k_fold cross validation.
    if k >= 3:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ddg_train, ddg_test = ddg[train_index], ddg[test_index]
            ## train model on each fold.

    ## blind test.
    if k == 2 and train_ratio > 0:
        positive_indices, negative_indices = ddg >= 0, ddg < 0
        x_positive, x_negative = x[positive_indices], x[negative_indices]
        y_positive, y_negative = y[positive_indices], y[negative_indices]
        ddg_positive, ddg_negative = ddg[positive_indices], ddg[negative_indices]
        left_positive = round(0.7 * x_positive.shape[0]); left_negative = round(0.7 * x_negative.shape[0])
        x_train, x_test = np.vstack((x_positive[:left_positive], x_negative[:left_negative])), np.vstack(
            (x_positive[left_positive:], x_negative[left_negative:]))
        y_train, y_test = np.vstack((y_positive[:left_positive], y_negative[:left_negative])), np.vstack(
            (y_positive[left_positive:], y_negative[left_negative:]))
        ddg_train, ddg_test = np.vstack((ddg_positive[:left_positive], ddg_negative[:left_negative])), np.vstack(
            (ddg_positive[left_positive:], ddg_negative[left_negative:]))
        x_train, y_train, ddg_train = shuffle_data(x_train, y_train, ddg_train, seed)
        x_test, y_test, ddg_test = shuffle_data(x_test, y_test, ddg_test, seed)
        ## train model.










    kfold_score = np.zeros((k, 16))  # 三列数据分别为：[测试数据总体准确率，正类测试数据正确率，负类测试数据准确率].
    # predicted_df = pd.DataFrame({'k_fold':[],'p0':[],'p1':[],'real_label':[]})
    k_count = 1
    history_list = []

        ## 开始训练和验证模型
        if nn_model == 0 or nn_model == 1 or nn_model == 1.01 or nn_model == 1.02 or nn_model == 1.03:
            ##分类模型
            [test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n, mcc,
             p_pred, y_real, history_dict, test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1,
             precision_n1, mcc1, p_pred1, y_real1] = train_model(
                dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train, ddg_train,
                x_test, y_test, ddg_test, k, k_count, nn_model=nn_model, normalize_method=normalize_method)

            history_list.append(history_dict)

            kfold_score[k_count - 1, :] = [test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p,
                                           precision_n, mcc,
                                           test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1,
                                           precision_n1, mcc1]

        elif nn_model == 2 or nn_model == 2.01 or nn_model == 2.02 or nn_model == 2.03:
            ##回归模型
            test_mse_score, test_mae_score, pearson_coeff, std, history_dict, test_mse_score1, test_mae_score1, pearson_coeff1, std1 = train_model(
                dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train, ddg_train,
                x_test, y_test, ddg_test, k, k_count, nn_model=nn_model, normalize_method=normalize_method)
            history_list.append(history_dict)
            kfold_score[k_count - 1,
            :8] = test_mse_score, test_mae_score, pearson_coeff, std, test_mse_score1, test_mae_score1, pearson_coeff1, std1

        k_count += 1

    return kfold_score, history_list

def blind_test(dataset_name, radius, k_neighbor, class_num, dist, x, y, ddg, nn_model=1, normalize_method=0):
    k_count = 1
    history_list = []
    kfold_score = np.zeros((k, 16))
    data_num, row_num, col_num = x.shape[0:3]
    train_num = int(data_num * 0.7)
    x_train, x_test = x[0:train_num], x[train_num:]
    y_train, y_test = y[0:train_num], y[train_num:]
    ddg_train, ddg_test = ddg[:train_num], ddg[train_num:]
    ## 开始训练和验证
    if nn_model == 0 or nn_model == 1 or nn_model == 1.01 or nn_model == 1.02 or nn_model == 1.03:  ##分类模型
        [test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n, mcc, p_pred, y_real,
         history_dict, test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1, precision_n1, mcc1,
         p_pred1, y_real1] = train_model(
            dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train, ddg_train, x_test, y_test, ddg_test, k,
            k_count, nn_model=nn_model, normalize_method=normalize_method)
        history_list.append(history_dict)
        kfold_score[k_count - 1, :] = [test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n,
                                       mcc, test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1,
                                       precision_n1, mcc1]

    elif nn_model == 2 or nn_model == 2.01 or nn_model == 2.02 or nn_model == 2.03:
        ##回归模型
        test_mse_score, test_mae_score, pearson_coeff, std, history_dict, test_mse_score1, test_mae_score1, pearson_coeff1, std1 = train_model(
            dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train, ddg_train, x_test, y_test, ddg_test, k,
            k_count, nn_model=nn_model, normalize_method=normalize_method)
        history_list.append(history_dict)
        kfold_score[k_count - 1,
        :8] = test_mse_score, test_mae_score, pearson_coeff, std, test_mse_score1, test_mae_score1, pearson_coeff1, std1

    return kfold_score, history_list


def kfold(dataset_name, radius, k_neighbor, class_num, dist, x, y, ddg, k=5, nn_model=1, normalize_method=0):
    """
    :param x: data_set array.
    :param y: data_set label array.
    :param k: k fold, default = 5.
    :nn_model: model to selected, 0-dense, 1-CNN, default = 1
    :return: k fold acc or other metrics.
    """
