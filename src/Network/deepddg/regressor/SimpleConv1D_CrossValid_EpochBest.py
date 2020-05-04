import os, json
import sys
import time

import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks
from mCNN.Network.metrics import test_report_reg, pearson_r, rmse
from keras.utils import to_categorical
from matplotlib import pyplot as plt

def data(train_data_pth,test_data_pth, val_data_pth):
    ## train data
    train_data = np.load(train_data_pth)
    x_train = train_data['x']
    y_train = train_data['y']
    ddg_train = train_data['ddg'].reshape(-1)
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    # class_weights_dict = dict(enumerate(class_weights))
    ## valid data
    val_data = np.load(val_data_pth)
    x_val = val_data['x']
    y_val = val_data['y']
    ddg_val = val_data['ddg'].reshape(-1)

    x_train = np.vstack((x_train,x_val))
    y_train = np.vstack((y_train,y_val))
    ddg_train = np.hstack((ddg_train,ddg_val))
    ## test data
    test_data = np.load(test_data_pth)
    x_test = test_data['x']
    y_test = test_data['y']
    ddg_test = test_data['ddg'].reshape(-1)

    # sort row default is chain, pass
    # reshape and one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # normalization
    train_shape = x_train.shape
    test_shape = x_test.shape
    col_train = train_shape[-1]
    col_test = test_shape[-1]
    x_train = x_train.reshape((-1, col_train))
    x_test = x_test.reshape((-1, col_test))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[np.argwhere(std == 0)] = 0.01
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    x_train = x_train.reshape(train_shape)
    x_test = x_test.reshape(test_shape)

    # reshape
    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, y_train, ddg_train, x_test, y_test, ddg_test

def ieee_net(x_train, y_train, ddg_train, epoch_best):
    row_num, col_num = x_train.shape[1:3]
    verbose = 1
    batch_size = 64
    epochs = epoch_best #[15, 12, 16, 29, 16, 12, 10, 31, 10, 19]

    metrics = ('mae', pearson_r, rmse)

    def step_decay(epoch):
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.00001
        elif epoch > 0.75 * epochs:
            lr = 0.0001
        elif epoch > 0.5 * epochs:
            lr = 0.001
        else:
            lr = 0.01
        print('lr: %f' % lr)
        return lr

    lrate = callbacks.LearningRateScheduler(step_decay, verbose=verbose)
    my_callbacks = [
        lrate
    ]

    network = models.Sequential()
    network.add(layers.Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(row_num, col_num)))
    network.add(layers.MaxPooling1D(pool_size=2))
    network.add(layers.Conv1D(32, 5, activation='relu'))
    network.add(layers.MaxPooling1D(pool_size=2))
    network.add(layers.Conv1D(64, 3, activation='relu'))
    network.add(layers.MaxPooling1D(pool_size=2))
    network.add(layers.Flatten())
    network.add(layers.Dense(128, activation='relu'))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dropout(0.3))
    network.add(layers.Dense(1))
    # print(network.summary())
    # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
    rmsp = optimizers.RMSprop(lr=0.0001)
    network.compile(optimizer=rmsp,#'rmsprop',  # SGD,adam,rmsprop
                    loss='mse',
                    metrics=list(metrics))  # mae平均绝对误差（mean absolute error） accuracy
    result = network.fit(x=x_train,
                         y=ddg_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         callbacks=my_callbacks,
                         shuffle=True,
                         )
    return network, result.history

if __name__ == '__main__':
    from mCNN.queueGPU import queueGPU
    CUDA_rate = '0.2'
    ## config TF
    queueGPU(USER_MEM=3000, INTERVAL=60)
    # os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate)<0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

    # modeldir = '/dl/sry/mCNN/src/Network/deepddg/regressor/TrySimpleConv1D_CrossValid_%s'%time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())
    modeldir = '/dl/sry/mCNN/src/Network/deepddg/regressor/%s_%s'%(sys.argv[0][:-3], time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()))
    os.makedirs(modeldir, exist_ok=True)
    score_dict = {'pearson_coeff':[], 'std':[], 'mae':[]}
    train_score_dict = {'pearson_coeff':[], 'std':[], 'mae':[]}
    es_train_score_dict = {'pearson_coeff':[], 'std':[], 'mae':[]}

    test_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_foldall_test_center_CA_PCA_False_neighbor_120.npz'
    for i in range(10):
        k_count = i+1
        fold_specific_epoch = [15, 12, 16, 29, 16, 12, 10, 31, 10, 19]
        print('--cross validation begin, fold %s is processing.'%k_count)
        train_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_fold%s_train_center_CA_PCA_False_neighbor_120.npz'%k_count
        valid_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_fold%s_valid_center_CA_PCA_False_neighbor_120.npz'%k_count
        x_train, y_train, ddg_train, x_test, y_test, ddg_test = data(train_data_pth,test_data_pth,valid_data_pth)
        print('x_train: %s'
              '\ny_train: %s'
              '\nddg_train: %s'
              '\nx_test: %s'
              '\ny_test: %s'
              '\nddg_test: %s'
              % (x_train.shape, y_train.shape, ddg_train.shape,
                 x_test.shape, y_test.shape, ddg_test.shape))

        #
        # train & test
        #
        model, history_dict = ieee_net(x_train, y_train, ddg_train,epoch_best=fold_specific_epoch[i])

        #
        # save model architecture
        #
        try:
            model_json = model.to_json()
            with open('%s/fold_%s_model.json'%(modeldir,k_count), 'w') as json_file:
                json_file.write(model_json)
        except:
            print('save model.json to json failed, fold_num: %s' % k_count)
        #
        # save model weights
        #
        try:
            model.save_weights(filepath='%s/fold_%s_weightsFinal.h5' % (modeldir,k_count))
        except:
            print('save final model weights failed, fold_num: %s' % k_count)
        #
        # save training history
        #
        try:
            with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'w') as file:
                file.write(str(history_dict))
            # with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'r') as file:
            #     print(eval(file.read()))
        except:
            print('save history_dict failed, fold_num: %s' % k_count)

        #
        # Load model
        #
        with open('%s/fold_%s_model.json'%(modeldir,k_count), 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
        loaded_model.load_weights(filepath='%s/fold_%s_weightsFinal.h5' % (modeldir,k_count))

        #
        # Test model
        #
        # pearson_coeff, std, mae = test_report_reg(model, x_test, ddg_test)
        pearson_coeff, std, mae = test_report_reg(loaded_model, x_test, ddg_test)
        print('\n----------Predict:'
              '\npearson_coeff: %s, std: %s, mae: %s'
              % (pearson_coeff, std, mae))
        score_dict['pearson_coeff'].append(pearson_coeff)
        score_dict['std'].append(std)
        score_dict['mae'].append(mae)

        train_score_dict['pearson_coeff'].append(history_dict['pearson_r'][-1])
        train_score_dict['std'].append(history_dict['rmse'][-1])
        train_score_dict['mae'].append(history_dict['mean_absolute_error'][-1])

        k_count += 1

    #
    # save score dict
    #
    try:
        with open('%s/fold_._score.dict' % modeldir, 'w') as file:
            file.write(str(score_dict))
    except:
        print('save score dict failed')

    #
    # save AVG score
    #
    try:
        with open('%s/fold_.avg_score_train_test.txt' % modeldir, 'w') as file:
            file.writelines('----------train AVG results\n')
            for key in score_dict.keys():
                file.writelines('*avg(%s): %s\n'%(key,np.mean(train_score_dict[key])))
            file.writelines('----------test AVG results\n')
            for key in score_dict.keys():
                file.writelines('*avg(%s): %s\n'%(key,np.mean(score_dict[key])))
    except:
        print('save AVG score failed')

    print('\nAVG results','-'*10)
    for key in score_dict.keys():
        print('*avg(%s): %s'%(key,np.mean(score_dict[key])))
