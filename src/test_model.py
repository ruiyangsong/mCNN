#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : test_model.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import time
import numpy as np
import scipy.stats as stats
from processing import split_delta_r


def test_model(network,x_test, y_test, ddg_test, nn_model):
    ## test
    print('testing %s model ...' % nn_model)
    if nn_model < 2:
        # test_loss, test_acc = network.evaluate(x_test, y_test)
        ## Calc evaluation metrics.
        p_pred = network.predict(x_test, batch_size=32, verbose=0) # Probability of test data belongs to each class, ndarray.
        y_pred = np.argmax(p_pred, axis=1)  # 0D array
        y_real = np.argmax(y_test, axis=1)
        # print(y_real.shape) # 1D
        # print(y_pred.shape) # 1D
        tp, fp, tn, fn = 0, 0, 0, 0
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
        recall_p, recall_n = tp/(tp+fn), tn/(tn+fp)
        precision_p, precision_n = tp/(tp+fp), tn/(tn+fn)
        mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        # print(recall_p,recall_n,precision_p)
        #report = classification_report(np.argmax(y_val, axis=1), y_pred)
        #print(type(report))  # str
        return test_acc, recall_p, recall_n, precision_p, precision_n, mcc

    elif nn_model > 2:
        #test_mse_score, test_mae_score = network.evaluate(x_test, ddg_test)
        #print('test_mse_score:%f,test_mae_score:%f'%(test_mse_score, test_mae_score))
        ## Clac pearson_coeff and rmse.
        ddg_pred = network.predict(x_test, batch_size=32, verbose=0)  # 测试数据的ddg值
        ddg_pred = ddg_pred.reshape(-1)
        ## save ddg_real ad ddg_pre to npz array.
        # file_name = 'nn_model' + str(nn_model) +'_'+ time.strftime("%Y%m%d%H%M%S", time.localtime())
        # np.savez('./%s.npz'%file_name, ddg_real=ddg_test, ddg_pred=ddg_pred)

        pearson_coeff,p_value = stats.pearsonr(ddg_test,ddg_pred)
        rmse = np.sum((ddg_test-ddg_pred)**2)/(len(ddg_test)-2)
        return pearson_coeff, rmse