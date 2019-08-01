#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : show_result.py
# time      : 4/6/2019 14:28
# author    : ruiyang
# email     : ww_sry@163.com
# ------------------------------

import numpy as np
import matplotlib.pyplot as plt

def print_result(nn_model,kfold_score,history_dict=0):
    if nn_model == 0 or nn_model == 1 or nn_model == 1.01 or nn_model == 1.02 or nn_model == 1.03:
        # print(history_dict)
        print('-' * 10, '验证结果如下列出：')
        #print('-' * 5, 'kfold_score:\n', kfold_score)
        print('-' * 5, '总验证数据k折平均准确率:', np.mean(kfold_score[:, 0]))
        #print('-' * 5, '正类测试数据k折平均准确率:', np.mean(kfold_score[:, 1]))
        #print('-' * 5, '负类测试数据k折平均准确率:', np.mean(kfold_score[:, 2]))
        print('-' * 5, 'recall_p的k折平均:', np.mean(kfold_score[:, 3]))
        print('-' * 5, 'recall_n的k折平均:', np.mean(kfold_score[:, 4]))
        print('-' * 5, 'precision_p的k折平均:', np.mean(kfold_score[:, 5]))
        print('-' * 5, 'precision_n的k折平均:', np.mean(kfold_score[:, 6]))
        print('-' * 5, 'mcc的k折平均:', np.mean(kfold_score[:, 7]))
        print('-' * 10, '预测结果如下列出：')
        #print('-' * 5, 'kfold_score:\n', kfold_score)
        print('-' * 5, '总测试数据k折平均准确率:', np.mean(kfold_score[:, 8]))
        #print('-' * 5, '正类测试数据k折平均准确率:', np.mean(kfold_score[:, 9]))
        #print('-' * 5, '负类测试数据k折平均准确率:', np.mean(kfold_score[:, 10]))
        print('-' * 5, 'recall_p的k折平均:', np.mean(kfold_score[:, 11]))
        print('-' * 5, 'recall_n的k折平均:', np.mean(kfold_score[:, 12]))
        print('-' * 5, 'precision_p的k折平均:', np.mean(kfold_score[:, 13]))
        print('-' * 5, 'precision_n的k折平均:', np.mean(kfold_score[:, 14]))
        print('-' * 5, 'mcc的k折平均:', np.mean(kfold_score[:, 15]))

    elif nn_model == 2 or nn_model == 2.01 or nn_model == 2.02 or nn_model == 2.03:
        # print(history_dict)
        print('-' * 10, '验证结果如下列出：')
        #print('-' * 5, 'kfold_score:\n', kfold_score)
        #print('-' * 5, 'k折均方误差:', np.mean(kfold_score[:, 0]))
        #print('-' * 5, 'k折平均绝对误差:', np.mean(kfold_score[:, 1]))
        print('-' * 5, '平均相关系数：', np.mean(kfold_score[:, 2]))
        print('-' * 5, '平均标准差：', np.mean(kfold_score[:, 3]))

        print('-' * 10, '测试结果如下列出：')
        #print('-' * 5, 'kfold_score:\n', kfold_score)
        #print('-' * 5, 'k折均方误差:', np.mean(kfold_score[:, 4]))
        #print('-' * 5, 'k折平均绝对误差:', np.mean(kfold_score[:, 5]))
        print('-' * 5, '平均相关系数：', np.mean(kfold_score[:, 6]))
        print('-' * 5, '平均标准差：', np.mean(kfold_score[:, 7]))


def plotfigure(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
