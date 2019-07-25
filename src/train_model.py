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
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from build_model import build_model
from sampling import oversampling, undersampling
from test_model import test_model, save_model
from show_result import print_result, plotfigure
from read_array import load_data
from processing import normalize, split_val, reshape_tensor, multi_channel, octant
from shuffle_data import sort_atom

def blind_test(dataset_name,radius, k_neighbor, class_num, dist,x, y,ddg,nn_model=1,normalize_method=0):
    k_count=1
    history_list=[]
    kfold_score = np.zeros((k,16))
    data_num,row_num,col_num = x.shape[0:3]
    train_num = int(data_num * 0.7)
    x_train, x_test = x[0:train_num], x[train_num:]
    y_train, y_test = y[0:train_num], y[train_num:]
    ddg_train, ddg_test = ddg[:train_num], ddg[train_num:]
    ## 开始训练和验证
    if nn_model == 0 or nn_model == 1 or nn_model == 1.01 or nn_model == 1.02 or nn_model == 1.03:                                                                            ##分类模型
        [test_acc, test_acc_p, test_acc_n,recall_p,recall_n,precision_p,precision_n,mcc,p_pred, y_real, history_dict,test_acc1, test_acc_p1, test_acc_n1,recall_p1,recall_n1,precision_p1,precision_n1,mcc1,p_pred1, y_real1] = train_model(
                dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train,ddg_train,x_test, y_test, ddg_test, k, k_count, nn_model=nn_model, normalize_method=normalize_method)
        history_list.append(history_dict)
        kfold_score[k_count-1, :] = [test_acc, test_acc_p, test_acc_n,recall_p,recall_n,precision_p,precision_n,mcc,test_acc1, test_acc_p1, test_acc_n1,recall_p1,recall_n1,precision_p1,precision_n1,mcc1]

    elif nn_model == 2 or nn_model == 2.01 or nn_model == 2.02 or nn_model == 2.03:
         ##回归模型
         test_mse_score,test_mae_score,pearson_coeff,std,history_dict,test_mse_score1,test_mae_score1,pearson_coeff1,std1 = train_model(
                 dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train,ddg_train,x_test, y_test, ddg_test,k, k_count, nn_model=nn_model, normalize_method=normalize_method)
         history_list.append(history_dict)
         kfold_score[k_count - 1, :8] = test_mse_score,test_mae_score,pearson_coeff,std,test_mse_score1,test_mae_score1,pearson_coeff1,std1

    return kfold_score, history_list


def kfold(dataset_name,radius, k_neighbor, class_num, dist,x, y,ddg, k = 5,nn_model=1,normalize_method=0):
    """
    :param x: data_set array.
    :param y: data_set label array.
    :param k: k fold, default = 5.
    :nn_model: model to selected, 0-dense, 1-CNN, default = 1
    :return: k fold acc or other metrics.
    """
    ## 计算k折数据
    skf = StratifiedKFold(n_splits = k, shuffle = False)
    kfold_score = np.zeros((k,16)) #三列数据分别为：[测试数据总体准确率，正类测试数据正确率，负类测试数据准确率].
    #predicted_df = pd.DataFrame({'k_fold':[],'p0':[],'p1':[],'real_label':[]})
    k_count = 1
    history_list = []
    for train_index, test_index in skf.split(x, y):
        ##切分每折的训练和测试集
        #print('-'*10,'正在进行第 %d 折...'%k_count)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ddg_train,ddg_test = ddg[train_index], ddg[test_index]
        ## 开始训练和验证模型
        if nn_model == 0 or nn_model == 1 or nn_model == 1.01 or nn_model == 1.02 or nn_model == 1.03:
            ##分类模型
            [test_acc, test_acc_p, test_acc_n,recall_p,recall_n,precision_p,precision_n,mcc,
             p_pred, y_real, history_dict,test_acc1, test_acc_p1, test_acc_n1,recall_p1,recall_n1,precision_p1,
             precision_n1,mcc1,p_pred1, y_real1] = train_model(
                dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train,ddg_train,
                x_test, y_test, ddg_test, k, k_count, nn_model=nn_model, normalize_method=normalize_method)

            history_list.append(history_dict)

            kfold_score[k_count-1, :] = [test_acc, test_acc_p, test_acc_n,recall_p,recall_n,precision_p,precision_n,mcc,
                                         test_acc1, test_acc_p1, test_acc_n1,recall_p1,recall_n1,precision_p1,precision_n1,mcc1]
        
        elif nn_model == 2 or nn_model == 2.01 or nn_model == 2.02 or nn_model == 2.03:
            ##回归模型
            test_mse_score,test_mae_score,pearson_coeff,std,history_dict,test_mse_score1,test_mae_score1,pearson_coeff1,std1 = train_model(
                dataset_name, radius, k_neighbor, class_num, dist, x_train, y_train,ddg_train,
                x_test, y_test, ddg_test,k, k_count, nn_model=nn_model, normalize_method=normalize_method)
            history_list.append(history_dict)
            kfold_score[k_count - 1, :8] = test_mse_score,test_mae_score,pearson_coeff,std,test_mse_score1,test_mae_score1,pearson_coeff1,std1

        k_count += 1

    return kfold_score, history_list

def train_model(dataset_name, radius, k_neighbor, class_num, dist,x_train, y_train,ddg_train, x_test, y_test, ddg_test,
                k, k_count, nn_model,normalize_method):
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
    ## 从训练集中切分出验证集
    x_train,y_train,ddg_train,x_val,y_val,ddg_val = split_val(x_train,y_train,ddg_train,x_test,y_test,k)
    ## 过采样
    if nn_model ==0 or nn_model == 1 or nn_model == 1.01 or nn_model== 1.02 or nn_model == 1.03:
        # 回归模型没有进行过采样！！！
        x_train, y_train = oversampling(x_train, y_train)
    ## 标准化
    x_train, x_val, x_test = normalize(x_train, x_val, x_test, normalize_method)
    ## 对数据reshape,使其符合相应的网络输入.
    x_train, x_test, x_val = reshape_tensor(x_train, x_test, x_val, nn_model)

    ## 单独选出正类测试集和负类测试集
    x_test_p = x_test[y_test.reshape(-1, ) == 1]
    y_test_p = y_test[y_test.reshape(-1, ) == 1]
    ddg_test_p = ddg_test[y_test.reshape(-1, ) == 1]
    x_val_p = x_val[y_val.reshape(-1, ) == 1]
    y_val_p = y_val[y_val.reshape(-1, ) == 1]
    ddg_val_p = ddg_val[y_test.reshape(-1, ) == 1]
    x_test_n = x_test[y_test.reshape(-1, ) == 0]
    y_test_n = y_test[y_test.reshape(-1, ) == 0]
    ddg_test_n = ddg_test[y_test.reshape(-1, ) == 0]
    x_val_n = x_val[y_val.reshape(-1, ) == 0]
    y_val_n = y_val[y_val.reshape(-1, ) == 0]
    ddg_val_n = ddg_val[y_test.reshape(-1, ) == 0]

    ## 进行类别编码独热编码.不影响ddg,ddg没有进行独热编码！
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)
    y_test_p = to_categorical(y_test_p, 2)
    y_test_n = to_categorical(y_test_n, 2)  # 当label全是0时，必须制定class_num才能正确编码为 one-hot 编码
    y_val_p = to_categorical(y_val_p, 2)
    y_val_n = to_categorical(y_val_n, 2)  # 当label全是0时，必须制定class_num才能正确编码为 one-hot 编码

    ## 实例化网络模型.
    sample_size = x_train.shape[1:3]#高度和宽度
    #print('x_train_shape:%r,y_train_shape:%r,x_val_shape:%r,y_val shape:%r'%(x_train.shape,y_train.shape,x_val.shape,y_val.shape))
    network = build_model(nn_model, sample_size)

    ## =======================================================================
    ## ----------------------------- train -----------------------------------
    ## =======================================================================
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, mode='auto')
    if nn_model ==1 or nn_model == 0:
        history = network.fit(
            x_train, y_train, validation_data=(x_val,y_val),
            epochs=100, batch_size=64, verbose=0,shuffle=True) #verbose=0 静默训练
        history_dict = history.history

    if nn_model == 1.03:
        x_train = x_train[:,:,:-5]
        x_train = x_train[:, :, :, np.newaxis]
        x_val = x_val[:, :, :-5]
        x_val = x_val[:, :, :, np.newaxis]
        
        history = network.fit(
            x_train, y_train, validation_data=(x_val,y_val),
            epochs=100, batch_size=64, verbose=0,shuffle=True) #verbose=0 静默训练
        history_dict = history.history

    elif nn_model == 1.01:
        ## 对x_train和x_val进行多通道表示
        x_train, pchange_train = multi_channel(x_train)
        x_val, pchange_val = multi_channel(x_val)

        history = network.fit(
            {'structure':x_train, 'pchange':pchange_train}, y_train,
            validation_data=([x_val,pchange_val], y_val),
            epochs=150, batch_size=64, verbose=0, shuffle=True)  # verbose=0 静默训练
        history_dict = history.history

    elif nn_model == 1.02:
        x_train, pchange_train = x_train[:,:,:-5], x_train[:,0,-5:]
        x_train = x_train[:, :, :, np.newaxis]
        x_val, pchange_val = x_val[:, :, :-5], x_val[:, 0, -5:]
        x_val = x_val[:, :, :, np.newaxis]
        history = network.fit(
            {'structure': x_train, 'pchange': pchange_train}, y_train,
            validation_data=([x_val, pchange_val], y_val),
            epochs=150, batch_size=64, verbose=0, shuffle=True)  # verbose=0 静默训练
        history_dict = history.history

    elif nn_model == 2:
        history = network.fit(
            x_train, ddg_train,validation_data=(x_val,ddg_val),
            epochs=100, batch_size=64, verbose=0)
        history_dict = history.history
    
    elif nn_model == 2.03:
        x_train = x_train[:,:,:-5]
        x_train = x_train[:, :, :, np.newaxis]
        x_val = x_val[:, :, :-5]
        x_val = x_val[:, :, :, np.newaxis]

        history = network.fit(
            x_train, ddg_train,validation_data=(x_val,ddg_val),
            epochs=100, batch_size=64, verbose=0)
        history_dict = history.history
        
    elif nn_model == 2.01:
        x_train, pchange_train = multi_channel(x_train)
        x_val, pchange_val = multi_channel(x_val)
        history = network.fit(
            {'structure':x_train, 'pchange':pchange_train}, ddg_train,
            validation_data=([x_val,pchange_val], ddg_val),
            epochs=150, batch_size=64, verbose=0)
        history_dict = history.history

    elif nn_model == 2.02:
        
        x_train, pchange_train = x_train[:,:,:-5], x_train[:,0,-5:]
        x_train = x_train[:, :, :, np.newaxis]
        x_val, pchange_val = x_val[:, :, :-5], x_val[:, 0, -5:]
        x_val = x_val[:, :, :, np.newaxis]
        history = network.fit(
            {'structure': x_train, 'pchange': pchange_train}, ddg_train,
            validation_data=([x_val, pchange_val], ddg_val),
            epochs=150, batch_size=64, verbose=0, shuffle=True)  # verbose=0 静默训练
        history_dict = history.history

    ## =======================================================================
    ## -----------------------------test--------------------------------------
    ## =======================================================================
    if nn_model == 0 or nn_model ==1:
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


    if nn_model == 1.03:
        x_test = x_test[:,:,:-5]
        x_test = x_test[:, :, :, np.newaxis]
        x_test_p = x_test_p[:,:,:-5]
        x_test_p = x_test_p[:, :, :, np.newaxis]

        x_test_n = x_test_n[:,:,:-5]
        x_test_n = x_test_n[:, :, :, np.newaxis]
        x_val_p = x_val_p[:,:,:-5]
        x_val_p = x_val_p[:, :, :, np.newaxis]
        x_val_n = x_val_n[:,:,:-5]
        x_val_n = x_val_n[:, :, :, np.newaxis]

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

    ## test
    elif nn_model == 1.01:
        x_test, pchange_test = multi_channel(x_test)
        x_test_p, pchange_test_p = multi_channel(x_test_p)
        x_test_n, pchange_test_n = multi_channel(x_test_n)
        x_val_p, pchange_val_p = multi_channel(x_val_p)
        x_val_n, pchange_val_n = multi_channel(x_val_n)

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
    elif nn_model == 2:
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

    
    elif nn_model == 2.03:
        x_test = x_test[:,:,:-5]
        x_test = x_test[:, :, :, np.newaxis]
        x_test_p = x_test_p[:,:,:-5]
        x_test_p = x_test_p[:, :, :, np.newaxis]
        x_test_n = x_test_n[:,:,:-5]
        x_test_n = x_test_n[:, :, :, np.newaxis]
        x_val_p = x_val_p[:,:,:-5]
        x_val_p = x_val_p[:, :, :, np.newaxis]
        x_val_n = x_val_n[:,:,:-5]
        x_val_n = x_val_n[:, :, :, np.newaxis]

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


    elif nn_model == 2.01:
        x_test, pchange_test = multi_channel(x_test)
        x_test_p, pchange_test_p = multi_channel(x_test_p)
        x_test_n, pchange_test_n = multi_channel(x_test_n)
        x_val_p, pchange_val_p = multi_channel(x_val_p)
        x_val_n, pchange_val_n = multi_channel(x_val_n)

        test_mse_score1, test_mae_score1, pearson_coeff1, std1 = test_model(
            network,
            [x_test, pchange_test], y_test, ddg_test,
            x_test_p, y_test_p,
            x_test_n, y_test_n,
            nn_model)

        test_mse_score, test_mae_score, pearson_coeff, std = test_model(
            network,
            [x_val,pchange_val], y_val, ddg_val,
            x_test_p, y_test_p,
            x_test_n, y_test_n,
            nn_model)

        return test_mse_score, test_mae_score, pearson_coeff, std, history_dict,test_mse_score1, test_mae_score1, pearson_coeff1, std1

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
    ## 初始化参数
    dataset_name, radius, k_neighbor, class_num, dist, k, nn_model, normalize_method, sort_value= sys.argv[1:]
    radius = float(radius) # 邻域半径
    k_neighbor = int(k_neighbor) # k 近邻数量
    class_num = int(class_num) # 原子类别数
    dist = int(dist) # 0 or 1
    k = int(k) # kfold
    nn_model = float(nn_model) # CNN or dense model
    normalize_method = int(normalize_method)

    # under_threshold = 20 #欠采样时考虑的不均衡比的阈值.
    ## load data
    x,y,ddg = load_data(dataset_name,radius,k_neighbor,class_num,dist,nn_model=nn_model)

    if int(sort_value) == 1:
        x = octant(x)
    elif int(sort_value) == 2:
        x = sort_atom(x,2)
    # ## 欠采样(阈值under_threshold = 20取得很大，不会进行欠采样)
    # x,y = undersampling(x,y,under_threshold)

    ## 交叉验证
    #print('-' * 10, '开始进行%d折交叉验证...'%k)
    if k == 1:
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
