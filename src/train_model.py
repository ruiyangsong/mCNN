#!～/anaconda3/env/bioinfo/bin/python
# -*- coding: utf-8 -*-

# file_name : train_model.py
# time      : 3/13/2019 13:52
# author    : ruiyang
# email     : ww_sry@163.com

import os
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from build_model import build_model
from processing import normalize, split_val, oversampling, reshape_tensor, split_delta_r

def train_model(x_train, y_train, ddg_train, x_test, y_test, ddg_test, nn_model, normalize_method, v_seed, flag_tuple, oversample,CUDA, epoch, batch_size):
    #############################################
    # ----------set train params here---------- #
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    #############################################
    ## processing data input.
    val_flag, verbose_flag = flag_tuple
    if val_flag == 0:
        verbose_flag = 0
    if val_flag == 1:
        ## Split val data from train data
        x_train, y_train, ddg_train, x_val, y_val, ddg_val = split_val(x_train,y_train,ddg_train,ddg_test, v_seed)
        # print('split val done.')
    elif val_flag == 0:
        x_val, y_val, ddg_val = np.array([]), np.array([]), np.array([])

    ## OverSampling train data for "classification" task.
    if nn_model < 2 and oversample == True:
        x_train, y_train = oversampling(x_train, y_train)
        # print('oversampling done.')

    ## Normalization.
    if val_flag == 1:
        x_train, x_test, x_val = normalize(x_train, x_test, x_val, val_flag, normalize_method)
    elif val_flag == 0:
        x_train, x_test = normalize(x_train, x_test, x_val, val_flag, normalize_method)
    # print('normalize done, normalize_method is %s.' % normalize_method)

    ## OneHot encoding for labels. Warnning: when labels are the same value, have to assign class number.
    classes = len(set(y_train.reshape(-1)))
    y_train, y_test= to_categorical(y_train, classes), to_categorical(y_test, classes)
    if val_flag == 1:
        y_val =  to_categorical(y_val, classes)

    ## Build CNN model.
    sample_size = x_train.shape[1:] # row_num and col_num
    #print('x_train_shape:%r,y_train_shape:%r,x_val_shape:%r,y_val shape:%r'%(x_train.shape,y_train.shape,x_val.shape,y_val.shape))
    network = build_model(nn_model, sample_size)
    ## =======================================================================
    ## ----------------------------- train -----------------------------------
    ## =======================================================================
    # print('training %s model ...' % nn_model)

    if nn_model ==1.01:
        ## Add axis for network input.
        x_train, x_test = reshape_tensor(x_train), reshape_tensor(x_test)
        if val_flag == 1:
            x_val = reshape_tensor(x_val)
            history = network.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        elif val_flag == 0:
            history = network.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 1.02:
        x_train, delta_r_train = split_delta_r(x_train)
        x_test, delta_r_test = split_delta_r(x_test)
        x_test = [x_test, delta_r_test]
        if val_flag == 1:
            x_val, delta_r_val = split_delta_r(x_val)
            history = network.fit({'structure': x_train, 'delta_r': delta_r_train}, y_train,
                                  validation_data=([x_val, delta_r_val], y_val),
                                  epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        elif val_flag == 0:
            history = network.fit({'structure': x_train, 'delta_r': delta_r_train}, y_train,
                                  epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 1.03:
        x_train, x_test = reshape_tensor(x_train), reshape_tensor(x_test)
        if val_flag == 1:
            x_val = reshape_tensor(x_val)
            history = network.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        elif val_flag == 0:
            history = network.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 2.01:
        ## Add axis for network input.
        x_train, x_test = reshape_tensor(x_train), reshape_tensor(x_test)
        if val_flag == 1:
            x_val = reshape_tensor(x_val)
            history = network.fit(x_train, ddg_train, validation_data=(x_val, ddg_val), epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        elif val_flag == 0:
            history = network.fit(x_train, ddg_train, epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 2.02:
        x_train, delta_r_train = split_delta_r(x_train)
        x_test, delta_r_test = split_delta_r(x_test)
        delta_r_train,delta_r_test = reshape_tensor(delta_r_train),reshape_tensor(delta_r_test)
        x_test = [x_test, delta_r_test]
        if val_flag == 1:
            x_val, delta_r_val = split_delta_r(x_val)
            history = network.fit({'mCNN': x_train, 'mCSM': delta_r_train}, ddg_train,
                                  validation_data=([x_val, delta_r_val], ddg_val),
                                  epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        elif val_flag == 0:
            history = network.fit({'mCNN': x_train, 'mCSM': delta_r_train}, ddg_train,
                                  epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    elif nn_model == 2.03:
        ## Add axis for network input.
        x_train, x_test = reshape_tensor(x_train), reshape_tensor(x_test)
        if val_flag == 1:
            x_val = reshape_tensor(x_val)
            history = network.fit(x_train, ddg_train, validation_data=(x_val, ddg_val), epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        elif val_flag == 0:
            history = network.fit(x_train, ddg_train, epochs=epoch, batch_size=batch_size, verbose=verbose_flag, shuffle=True)
        history_dict = history.history

    return network, history_dict, x_test, y_test, ddg_test

if __name__ == '__main__':
    pass
