import os, json
import sys
import time

import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks,regularizers
from mCNN.Network.metrics import test_report_reg, pearson_r, rmse
from mCNN.Network.lossFunction import logcosh

from matplotlib import pyplot as plt


def loss_plot(history_dict, outpth):
    loss = history_dict['loss']
    mean_absolute_error = history_dict['mean_absolute_error']
    pearson_r = history_dict['pearson_r']
    rmse = history_dict['rmse']
    val_loss = history_dict['val_loss']
    val_mean_absolute_error = history_dict['val_mean_absolute_error']
    val_pearson_r = history_dict['val_pearson_r']
    val_rmse = history_dict['val_rmse']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss (mse)')
    plt.plot(epochs, val_loss, 'g', label='Validation loss (mse)')
    # plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, mean_absolute_error, 'r', label='Training mae')
    plt.plot(epochs, val_mean_absolute_error, 'g', label='Validation mae')
    # plt.title('Training and validation mae')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, rmse, 'r', label='Training rmse')
    plt.plot(epochs, val_rmse, 'g', label='Validation rmse')
    # plt.title('Training and validation rmse')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, pearson_r, 'r', label='Training pcc')
    plt.plot(epochs, val_pearson_r, 'g', label='Validation pcc')
    # plt.title('Training and validation pcc')
    plt.xlabel('Epochs')
    plt.ylabel('PCC')
    plt.grid(True)
    plt.legend(loc="lower right")

    # plt.show()
    plt.savefig(outpth)#pngfile
    plt.clf()

def _data(train_data_pth, kneighbor, split_rate=0.7):
    from keras.utils import to_categorical
    ## train data
    train_data = np.load(train_data_pth)
    x_train = train_data['x']
    y_train = train_data['y']
    ddg_train = train_data['ddg'].reshape(-1)

    ## select kneighbor atoms
    x_train_kneighbor_lst = []
    for sample in x_train:
        dist_arr = sample[:,0]
        indices = sorted(dist_arr.argsort()[:kneighbor])
        x_train_kneighbor_lst.append(sample[indices, :])
    x_train = np.array(x_train_kneighbor_lst)



    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    # class_weights_dict = dict(enumerate(class_weights))
    # print(x_train.shape)

    ## valid data
    random_seed = 527#date20200527
    train_num = x_train.shape[0]
    indices = [i for i in range(train_num)]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    ddg_train = ddg_train[indices]
    positive_indices, negative_indices = ddg_train >= 0, ddg_train < 0
    x_positive, x_negative = x_train[positive_indices], x_train[negative_indices]
    y_positive, y_negative = y_train[positive_indices], y_train[negative_indices]
    ddg_positive, ddg_negative = ddg_train[positive_indices], ddg_train[negative_indices]

    left_positive, left_negative = round(split_rate * x_positive.shape[0]), round(split_rate * x_negative.shape[0])
    x_train = np.vstack((x_positive[:left_positive], x_negative[:left_negative]))
    x_val = np.vstack((x_positive[left_positive:], x_negative[left_negative:]))
    y_train = np.vstack((y_positive[:left_positive], y_negative[:left_negative]))
    y_val = np.vstack((y_positive[left_positive:], y_negative[left_negative:]))
    ddg_train = np.hstack((ddg_positive[:left_positive], ddg_negative[:left_negative]))
    ddg_val = np.hstack((ddg_positive[left_positive:], ddg_negative[left_negative:]))


    # sort row default is chain, pass
    # reshape and one-hot
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    # normalization
    train_shape = x_train.shape
    val_shape = x_val.shape
    col_train = train_shape[-1]
    col_val = val_shape[-1]
    x_train = x_train.reshape((-1, col_train))
    x_val = x_val.reshape((-1, col_val))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[np.argwhere(std == 0)] = 0.01
    x_train -= mean
    x_train /= std
    x_val -= mean
    x_val /= std
    x_train = x_train.reshape(train_shape)
    x_val = x_val.reshape(val_shape)

    # reshape
    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))
    # x_val = x_val.reshape(x_val.shape + (1,))
    print('x_train: %s'
          '\ny_train: %s'
          '\nddg_train: %s'
          '\nx_val: %s'
          '\ny_val: %s'
          '\nddg_val: %s'
          % (x_train.shape, y_train.shape, ddg_train.shape,
             x_val.shape, y_val.shape, ddg_val.shape))
    return x_train, y_train, ddg_train, x_val, y_val, ddg_val
    # return x_train, y_train, ddg_train, x_val, y_val, ddg_val, x_val, y_val, ddg_val

def Conv1DRegressorIn1(x_train, y_train, ddg_train,  x_val, y_val, ddg_val,filepth):
    K.clear_session()
    summary = False
    verbose = 0

    #
    # setHyperParams
    #
    row_num, col_num = x_train.shape[1:3]
    batch_size = 32#{{choice([128, 32, 64])}}  # 128#64
    epochs = 200
    padding_style = 'same'

    activator_Conv1D = 'elu'
    activator_Dense = 'tanh'
    dense_num = 64  # 64
    dilation1D_layers = 8
    dilation1D_filter_num = 16
    dilation_lower = 1
    dilation_upper = 16
    dropout_rate_dilation = 0.25#{{uniform(0.1, 0.35)}}
    dropout_rate_reduce = 0.25#{{uniform(0.1, 0.25)}}
    dropout_rate_dense = 0.25

    initializer_Conv1D = 'lecun_uniform'
    initializer_Dense = 'he_normal'
    kernel_size = 9#{{choice([9, 5, 7, 3])}}
    l2_rate = 0.045#{{uniform(0.01, 0.35)}}
    loss_type = logcosh
    lr = 0.0001
    metrics = ('mae', pearson_r, rmse)

    my_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.33,
            patience=5,
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
        ),
        callbacks.ModelCheckpoint(
            filepath=filepth,
            # monitor='val_pearson_r',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True)
    ]
    pool_size = 2
    reduce_layers = 5
    reduce1D_filter_num = 32
    residual_stride = 2

    network = models.Sequential()
    network.add(layers.SeparableConv1D(filters=16, kernel_size=5, activation='relu', input_shape=(row_num, col_num)))
    network.add(layers.MaxPooling1D(pool_size=2))
    network.add(layers.SeparableConv1D(32, 5, activation='relu'))
    network.add(layers.MaxPooling1D(pool_size=2))
    network.add(layers.SeparableConv1D(64, 3, activation='relu'))
    network.add(layers.MaxPooling1D(pool_size=2))
    network.add(layers.Flatten())
    network.add(layers.Dense(128, activation='relu'))
    network.add(layers.Dropout(0.3))
    network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dropout(0.3))
    network.add(layers.Dense(1))
    # print(network.summary())
    # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
    rmsp = optimizers.RMSprop(lr=0.0001)
    network.compile(optimizer=rmsp,  # 'rmsprop',  # SGD,adam,rmsprop
                    loss='mae',
                    metrics=list(metrics))  # mae平均绝对误差（mean absolute error） accuracy
    result = network.fit(x=x_train,
                         y=ddg_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         callbacks=my_callbacks,
                         validation_data=(x_val, ddg_val),
                         shuffle=True,
                         )

    return model, result.history

def config_tf():
    from mCNN.queueGPU import queueGPU
    try:
        from mCNN.processing import check_pid
        pid = sys.argv[1]
        check_pid(pid)
    except:
        pass
    CUDA_rate = '0.25'
    ## config TF
    queueGPU(USER_MEM=3000, INTERVAL=60)
    # os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate) < 0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

def _train(modeldir,neighbor):
    os.makedirs(modeldir, exist_ok=True)
    score_dict = {'pearson_coeff': [], 'std': [], 'mae': []}
    train_score_dict = {'pearson_coeff': [], 'std': [], 'mae': []}
    es_train_score_dict = {'pearson_coeff': [], 'std': [], 'mae': []}

    train_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/train_data_neighbor_140.npz'
    x_train, y_train, ddg_train, x_val, y_val, ddg_val = _data(train_data_pth,neighbor, split_rate=0.7)

    #
    # train & test
    #
    # filepth = '%s/fold_%s_weights-improvement-{epoch:02d}-{val_loss:.4f}.h5'%(modeldir,k_count)
    filepth = '%s/weights-best.h5' % modeldir
    model, history_dict = Conv1DRegressorIn1(x_train, y_train, ddg_train, x_val, y_val, ddg_val, filepth)

    #
    # save model architecture
    #
    try:
        model_json = model.to_json()
        with open('%s/model.json' % modeldir, 'w') as json_file:
            json_file.write(model_json)
    except:
        print('save model.json to json failed')
    #
    # save model weights
    #
    try:
        model.save_weights(filepath='%s/weightsFinal.h5' % (modeldir))
    except:
        print('save final model weights failed')
    #
    # save training history
    #
    try:
        with open('%s/history.dict' % (modeldir), 'w') as file:
            file.write(str(history_dict))
        # with open('%s/fold_%s_history.dict'%(modeldir,k_count), 'r') as file:
        #     print(eval(file.read()))
    except:
        print('save history_dict failed, fold_num: %s')
    #
    # save loss figure
    #
    try:
        figure_pth = '%s/lossFigure.png' % (modeldir)
        loss_plot(history_dict, outpth=figure_pth)
    except:
        print('save loss plot figure failed')

if __name__ == '__main__':
    config_tf()
    neighbor_lst = [50, 60, 70, 80, 90, 100]
    for neighbor in neighbor_lst:
        print('\n--------------------------------------------------'
              '\n kneighbor: %s'
              '\n--------------------------------------------------'%neighbor)
        modeldir = '/dl/sry/projects/from_hp/mCNN/src/Network/deepddg/feature158/choseFeature/v4/model/%s_neighbor_%s'%(
            sys.argv[0][:-3], neighbor)
        _train(modeldir,neighbor)

