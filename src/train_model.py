#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : train_model.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import sys
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from build_model import build_model
from test_model import test_model, save_model
from show_result import print_result, plotfigure
from processing import load_data, sort_row, normalize, split_val, reshape_tensor, oversampling, split_delta_r

def train_model(x_train, y_train, ddg_train, x_test, y_test, ddg_test, seed, k, k_count, nn_model, normalize_method):
    """
    :note: dropout 应用于前面一层的输出.
    :param x_train: 训练集array.
    :param y_train: 训练集labels. so as x_test, y_test.
    :param nn_model: 选择的网络模型种类, nn_model = [0,1]
                     0 - 全连接网络 (as default)
                     1 - CNN       
    :return: return model, metrics
    """

    x_train = x_train.astype('float32')  # / 100
    x_test = x_test.astype('float32')  # / 100
    ## Split val data from train data
    x_train, y_train, ddg_train, x_val, y_val, ddg_val = split_val(x_train,y_train,ddg_train,ddg_test, seed)
    ## OverSampling train data for classification task.
    if nn_model < 2:
        x_train, y_train = oversampling(x_train, y_train)
    ## Normalization.
    x_train, x_test, x_val = normalize(x_train, x_test, x_val, normalize_method)

    # ## Add axis for network input.
    # x_train, x_test, x_val = reshape_tensor(x_train, x_test, x_val)

    ## Select positive and negative test data and val data.
    test_p_indices, test_n_indices = ddg_test >= 0, ddg_test < 0
    val_p_indices, val_n_indices = ddg_val >= 0, ddg_val < 0
    x_test_p, x_test_n = x_test[test_p_indices], x_test[test_n_indices]
    y_test_p, y_test_n = y_test[test_p_indices], y_test[test_n_indices]
    x_val_p, x_val_n = x_val[val_p_indices], x_val[val_n_indices]
    y_val_p, y_val_n = y_val[val_p_indices], y_val[val_n_indices]
    ## OneHot encoding for labels. Warnning: when labels are the same value, have to assign class number.
    y_train, y_test, y_val = to_categorical(y_train, 2), to_categorical(y_test, 2), to_categorical(y_val, 2)
    y_test_p, y_test_n = to_categorical(y_test_p, 2), to_categorical(y_test_n, 2)
    y_val_p, y_val_n = to_categorical(y_val_p, 2), to_categorical(y_val_n, 2)
    ## Build CNN model.
    sample_size = x_train.shape[1:3] # row_num and col_num
    #print('x_train_shape:%r,y_train_shape:%r,x_val_shape:%r,y_val shape:%r'%(x_train.shape,y_train.shape,x_val.shape,y_val.shape))
    network = build_model(nn_model, sample_size)
    ## =======================================================================
    ## ----------------------------- train -----------------------------------
    ## =======================================================================
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto')
    if nn_model ==1.01:
        history = network.fit(
            x_train, y_train, validation_data=(x_val,y_val),
            epochs=100, batch_size=64, verbose=0,shuffle=True)
        history_dict = history.history
    elif nn_model == 2.01:
        history = network.fit(
            x_train, ddg_train,validation_data=(x_val,ddg_val),
            epochs=100, batch_size=64, verbose=0,shuffle=True)
        history_dict = history.history

    elif nn_model == 1.02:
        x_train, delta_r_train = split_delta_r(x_train)
        x_val, delta_r_val = split_delta_r(x_val)

        history = network.fit(
            {'structure': x_train, 'delta_r': delta_r_train}, y_train,
            validation_data=([x_val, delta_r_val], y_val),
            epochs=150, batch_size=64, verbose=0, shuffle=True)  # verbose=0 静默训练
        history_dict = history.history

    elif nn_model == 2.02:
        x_train, delta_r_train = split_delta_r(x_train)
        x_val, delta_r_val = split_delta_r(x_val)
        history = network.fit(
            {'structure': x_train, 'delta_r': delta_r_train}, ddg_train,
            validation_data=([x_val, delta_r_val], ddg_val),
            epochs=150, batch_size=64, verbose=0, shuffle=True)  # verbose=0 静默训练
        history_dict = history.history

    ## =======================================================================
    ## -----------------------------test--------------------------------------
    ## =======================================================================
    if nn_model == 1.01:
        test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1, precision_n1, mcc1, p_pred1, y_real1 = test_model(
            network,
            x_test,y_test,ddg_test,
            x_test_p,y_test_p,
            x_test_n,y_test_n,
            nn_model)

        test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n, mcc, p_pred, y_real = test_model(
            network,
            x_val,y_val,ddg_val,
            x_val_p, y_val_p,
            x_val_n, y_val_n,
            nn_model)

        # ## 保存准确率高于0.86的模型
        # acc_threshold = 0.86
        # save_model(dataset_name, radius, k_neighbor, class_num, dist, network, test_acc, k_count,acc_threshold=acc_threshold)

        return [test_acc, test_acc_p, test_acc_n,recall_p,recall_n,precision_p,precision_n,mcc, p_pred, y_real, history_dict,
                test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1, precision_n1, mcc1, p_pred1, y_real1]

    elif nn_model == 1.02:
        x_test, pchange_test = x_test[:,:,:-5], x_test[:,0,-5:]
        x_test = x_test[:, :, :, np.newaxis]
        x_test_p, pchange_test_p = x_test_p[:,:,:-5], x_test_p[:,0,-5:]
        x_test_p = x_test_p[:, :, :, np.newaxis]
        x_test_n, pchange_test_n = x_test_n[:,:,:-5], x_test_n[:,0,-5:]
        x_test_n = x_test_n[:, :, :, np.newaxis]
        x_val_p, pchange_val_p = x_val_p[:,:,:-5], x_val_p[:,0,-5:]
        x_val_p = x_val_p[:, :, :, np.newaxis]
        x_val_n, pchange_val_n = x_val_n[:,:,:-5], x_val_n[:,0,-5:]
        x_val_n = x_val_n[:, :, :, np.newaxis]

        test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1, precision_n1, mcc1, p_pred1, y_real1 = test_model(
            network,
            [x_test,pchange_test], y_test, ddg_test,
            [x_test_p,pchange_test_p], y_test_p,
            [x_test_n,pchange_test_n], y_test_n,
            nn_model)

        test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n, mcc, p_pred, y_real = test_model(
            network,
            [x_val, pchange_val], y_val, ddg_val,
            [x_val_p, pchange_val_p], y_val_p,
            [x_val_n, pchange_val_n], y_val_n,
            nn_model)

        return [test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n, mcc, p_pred, y_real,history_dict,
                test_acc1, test_acc_p1, test_acc_n1, recall_p1, recall_n1, precision_p1, precision_n1, mcc1, p_pred1, y_real1]

    ## test
    elif nn_model == 2.01:
        test_mse_score1, test_mae_score1, pearson_coeff1, std1 = test_model(
            network,
            x_test, y_test, ddg_test,
            x_test_p, y_test_p,
            x_test_n, y_test_n,
            nn_model)

        test_mse_score, test_mae_score,pearson_coeff,std = test_model(
            network,
            x_val,y_val,ddg_val,
            x_test_p,y_test_p,
            x_test_n,y_test_n,
            nn_model)
        return test_mse_score, test_mae_score, pearson_coeff,std,history_dict,test_mse_score1, test_mae_score1, pearson_coeff1, std1

    elif nn_model == 2.02:
        x_test, pchange_test = x_test[:,:,:-5], x_test[:,0,-5:]
        x_test = x_test[:, :, :, np.newaxis]
        x_test_p, pchange_test_p = x_test_p[:,:,:-5], x_test_p[:,0,-5:]
        x_test_p = x_test_p[:, :, :, np.newaxis]
        x_test_n, pchange_test_n = x_test_n[:,:,:-5], x_test_n[:,0,-5:]
        x_test_n = x_test_n[:, :, :, np.newaxis]
        x_val_p, pchange_val_p = x_val_p[:,:,:-5], x_val_p[:,0,-5:]
        x_val_p = x_val_p[:, :, :, np.newaxis]
        x_val_n, pchange_val_n = x_val_n[:,:,:-5], x_val_n[:,0,-5:]
        x_val_n = x_val_n[:, :, :, np.newaxis]
        
        test_mse_score, test_mae_score, pearson_coeff, std = test_model(
            network,
            [x_test,pchange_test], y_test, ddg_test,
            x_test_p, y_test_p,
            x_test_n, y_test_n,
            nn_model)

        test_mse_score1, test_mae_score1, pearson_coeff1, std1 = test_model(
            network,
            [x_val, pchange_val], y_val, ddg_val,
            x_val_p, y_val_p,
            x_val_n, y_val_n,
            nn_model)

        return test_mse_score, test_mae_score, pearson_coeff, std, history_dict,test_mse_score1, test_mae_score1, pearson_coeff1, std1



if __name__ == '__main__':
    ## Input parameters.
    dataset_name, radius, k_neighbor, class_num, k, nn_model, standardization_method, sort_method= sys.argv[1:]
    radius = float(radius)
    k_neighbor = int(k_neighbor)
    class_num = int(class_num) # class number of atoms.
    k = int(k) # kfold
    nn_model = float(nn_model) # which CNN structure to choose.

    ## load data
    x, y, ddg = load_data(dataset_name,radius,k_neighbor,class_num)
    print('Loading data from hard drive is done.')

    ## sort row of each mutation matrix.
    x = sort_row(x, sort_method)
    print('Sort row is done, sorting method is %s.' % sort_method)

    ## Cross validation.
    if k == 2:
        kfold_score, history_dict = blind_test(
                dataset_name,radius,k_neighbor,class_num,dist,x,y,ddg,nn_model=nn_model,normalize_method = normalize_method)
    else:
        kfold_score, history_dict = kfold(
                dataset_name, radius, k_neighbor, class_num, dist, x, y,ddg, k, nn_model=nn_model, normalize_method = normalize_method)
    ## 打印交叉验证结果
    print_result(nn_model,kfold_score,history_dict)
    # print_result(nn_model, kfold_score)
    ## 画图
    #plotfigure(history_dict)
