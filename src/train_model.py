#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : train_model.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

from keras.utils import to_categorical
from build_model import build_model
from processing import normalize, split_val, oversampling, reshape_tensor, split_delta_r

def train_model(x_train, y_train, ddg_train, x_test, y_test, ddg_test, seed, nn_model, normalize_method):
    verbose_flag = 0
    ## set dtype of x_train and x_test.
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    ## Split val data from train data
    x_train, y_train, ddg_train, x_val, y_val, ddg_val = split_val(x_train,y_train,ddg_train,ddg_test, seed)
    # print('split val done.')

    ## OverSampling train data for classification task.
    if nn_model < 2:
        x_train, y_train = oversampling(x_train, y_train)
        # print('oversampling done.')

    ## Normalization.
    x_train, x_test, x_val = normalize(x_train, x_test, x_val, normalize_method)
    # print('normalize done, normalize_method is %s.' % normalize_method)

    ## OneHot encoding for labels. Warnning: when labels are the same value, have to assign class number.
    y_train, y_test, y_val = to_categorical(y_train, 2), to_categorical(y_test, 2), to_categorical(y_val, 2)

    ## Build CNN model.
    sample_size = x_train.shape[1:3] # row_num and col_num
    #print('x_train_shape:%r,y_train_shape:%r,x_val_shape:%r,y_val shape:%r'%(x_train.shape,y_train.shape,x_val.shape,y_val.shape))
    network = build_model(nn_model, sample_size)
    ## =======================================================================
    ## ----------------------------- train -----------------------------------
    ## =======================================================================
    # print('training %s model ...' % nn_model)
    if nn_model ==1.01:
        ## Add axis for network input.
        x_train, x_test, x_val = reshape_tensor(x_train, x_test, x_val)
        history = network.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            epochs=100, batch_size=64, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 2.01:
        ## Add axis for network input.
        x_train, x_test, x_val = reshape_tensor(x_train, x_test, x_val)
        history = network.fit(
            x_train, ddg_train, validation_data=(x_val, ddg_val),
            epochs=100, batch_size=64, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 1.02:
        x_train, delta_r_train = split_delta_r(x_train)
        x_val, delta_r_val = split_delta_r(x_val)
        x_test, delta_r_test = split_delta_r(x_test)
        x_test = [x_test, delta_r_test]

        history = network.fit(
            {'structure': x_train, 'delta_r': delta_r_train}, y_train,
            validation_data=([x_val, delta_r_val], y_val),
            epochs=150, batch_size=64, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 2.02:
        x_train, delta_r_train = split_delta_r(x_train)
        x_val, delta_r_val = split_delta_r(x_val)
        x_test, delta_r_test = split_delta_r(x_test)
        x_test = [x_test, delta_r_test]

        history = network.fit(
            {'structure': x_train, 'delta_r': delta_r_train}, ddg_train,
            validation_data=([x_val, delta_r_val], ddg_val),
            epochs=150, batch_size=64, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    return network, history_dict, x_test, y_test, ddg_test

if __name__ == '__main__':
    pass