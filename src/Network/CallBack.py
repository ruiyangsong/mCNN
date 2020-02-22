#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import matplotlib.pyplot as plt

class TrainCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.pearson_r = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_pearson_r = {'batch':[], 'epoch':[]}

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.pearson_r['epoch'].append(logs.get('pearson_r'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_pearson_r['epoch'].append(logs.get('val_pearson_r'))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.pearson_r['batch'].append(logs.get('pearson_r'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_pearson_r['batch'].append(logs.get('val_pearson_r'))


    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.pearson_r[loss_type], 'r', label='train_pearson_r')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train_loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_pearson_r[loss_type], 'b', label='val_pearson_r')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val_loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('pearson_r-loss')
        plt.legend(loc="upper right")
        plt.savefig("/public/home/yels/project/Data_pre/QA_global.png")
        # plt.show()