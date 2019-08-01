#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : test_model.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com
import os
import time
import numpy as np
import scipy.stats as stats


def test_model(network,x_test,y_test,ddg_test,x_test_p,y_test_p,x_test_n,y_test_n,nn_model):
    ## test
    #print('testing.', '_' * 10)
    if nn_model == 0 or nn_model == 1 or nn_model == 1.01 or nn_model == 1.02 or nn_model == 1.03:
        # test_loss, test_acc = network.evaluate(x_test, y_test)
        test_loss_p, test_acc_p = 0, 0
        test_loss_n, test_acc_n = 0, 0
        # test_loss_p, test_acc_p = network.evaluate(x_test_p, y_test_p)
        # test_loss_n, test_acc_n = network.evaluate(x_test_n, y_test_n)
        #print('test_acc:%f,test_acc_p:%f,test_acc_n:%f'%(test_acc,test_acc_p,test_acc_n))
        ##预测的recall和precision
        p_pred = network.predict(x_test, batch_size=32, verbose=0)  # 测试数据属于每一个类的概率,ndarray
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
        test_acc = (tp+tn)/(tp+tn+fp+fn)
        recall_p = tp/(tp+fn)
        recall_n = tn/(tn+fp)
        precision_p = tp/(tp+fp)
        precision_n = tn/(tn+fn)
        mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        # print(recall_p,recall_n,precision_p)
        #report = classification_report(np.argmax(y_val, axis=1), y_pred)
        #print(type(report))  # str
        return test_acc, test_acc_p, test_acc_n, recall_p, recall_n, precision_p, precision_n, mcc, p_pred, y_real

    elif nn_model == 2 or nn_model == 2.01 or nn_model == 2.02 or nn_model == 2.03:
        test_mse_score, test_mae_score = 0,0
        # test_mse_score, test_mae_score = network.evaluate(x_test, ddg_test)
        #print('test_mse_score:%f,test_mae_score:%f'%(test_mse_score, test_mae_score))
        ##计算ddg_pred 和 ddg_real（ddg_test）的相关系数，和标准差
        ddg_pred = network.predict(x_test, batch_size=32, verbose=0)  # 测试数据的ddg值
        ddg_pred = ddg_pred.reshape(-1)
        #print(ddg_pred,ddg_test)
        # ## save ddg_real ad ddg_pre to npz array.
        # file_name = 'nn_model' + str(nn_model) +'_'+ time.strftime("%Y%m%d%H%M%S", time.localtime())
        # np.savez('./%s.npz'%file_name, ddg_real=ddg_test, ddg_pred=ddg_pred)

        pearson_coeff,p_value = stats.pearsonr(ddg_test,ddg_pred)
        #std = np.std(ddg_test-ddg_pred)
        std = np.sum((ddg_test-ddg_pred)**2)/(len(ddg_test)-2)
        return test_mse_score, test_mae_score,pearson_coeff,std

def save_model(dataset_name, radius, k_neighbor, class_num, dist,network,test_acc,k_count,acc_threshold=0.86):
    ##创建目录
    path_dist = '../models/' + dataset_name + '/dist/'
    path_k_neighbor = '../models/' + dataset_name + '/k_neighbor/'
    path_radius = '../models/' + dataset_name + '/radius/'
    if not os.path.exists(path_dist):
        os.mkdir(path_dist)
    if not os.path.exists(path_k_neighbor):
        os.mkdir(path_k_neighbor)
    if not os.path.exists(path_radius):
        os.mkdir(path_radius)
    ##保存模型
    if test_acc >= acc_threshold:
        if dist == 1:
            #将模型存入dist文件夹
            network.save('../models/%s/dist/r_%.2f_neighbor_%d_class_%d_acc_%.4f_kcount_%d.h5' % (
                dataset_name, radius,k_neighbor,class_num,test_acc,k_count))
        elif k_neighbor != 0:
            #将模型存入k_neighbor文件夹
            network.save('../models/%s/k_neighbor/r_%.2f_neighbor_%d_class_%d_acc_%.4f_kcount_%d.h5' % (
                dataset_name, radius,k_neighbor,class_num,test_acc,k_count))
        else:
            #将模型存入radius文件夹
            network.save('../models/%s/radius/r_%.2f_neighbor_%d_class_%d_acc_%.4f_kcount_%d.h5' % (
                dataset_name, radius,k_neighbor,class_num,test_acc,k_count))
