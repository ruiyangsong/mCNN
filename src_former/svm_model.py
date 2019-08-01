#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : svm_model.py
# time      : 4/4/2019 17:17
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import sys
import numpy as np
from sklearn import svm
from train_model import load_data
from sklearn.model_selection import StratifiedKFold


def k_fold(k,x,y):
    print('svm_model')
    print('x shape:',x.shape)
    x = x.reshape(data_num, -1) # 将数据展成1D的向量
    print('x shape:',x.shape)
    skf = StratifiedKFold(n_splits = k, shuffle = False)
    kfold_score = np.zeros((k,8)) #三列数据分别为：[测试数据总体准确率，正类测试数据正确率，负类测试数据准确率].
    k_count = 1
    for train_index, test_index in skf.split(x, y):
        ##切分每折的训练和测试集
        print('-'*10,'正在进行第 %d 折...'%k_count)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('y_train shape:',y_train.shape)
        ##训练
        clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf.fit(x_train, y_train.ravel()) # 0D Tensor
        print (clf.score(x_train, y_train))  # 精度
        # y_hat = clf.predict(x_train)
        print (clf.score(x_test, y_test))
        y_hat = clf.predict(x_test)
        print(y_hat)
        print(y_test.reshape(-1))
        k_count+=1


if __name__ == '__main__':
    # dataset_name, radius, k_neighbor, class_num, dist, nn_model = sys.argv[1:]
    #     # radius = float(radius)  # 邻域半径
    #     # k_neighbor = int(k_neighbor)  # k 近邻数量
    #     # class_num = int(class_num)  # 原子类别数
    #     # dist = int(dist)  # 0 or 1
    dataset_name, radius, k_neighbor, class_num, dist = 'S1932',50.00,50,5,0
    x, y, ddg = load_data(dataset_name, radius, k_neighbor, class_num, dist, nn_model=1)
    data_num,row,col = x.shape

    k_fold(k=20,x=x,y=y)