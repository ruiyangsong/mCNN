'split train test randomly'
import os, json
import time

import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks
from mCNN.Network.metrics import test_report_reg, pearson_r
from mCNN.queueGPU import queueGPU
from keras.utils import to_categorical
from matplotlib import pyplot as plt

def loss_plot(history_dict,outpth):
    loss                = history_dict['loss']
    mean_absolute_error = history_dict['mean_absolute_error']
    pearson_r           = history_dict['pearson_r']
    val_loss                = history_dict['val_loss']
    val_mean_absolute_error = history_dict['val_mean_absolute_error']
    val_pearson_r           = history_dict['val_pearson_r']
    epochs = range(1, len(loss) + 1)
    plt.subplot(3,1,1)
    plt.plot(epochs, loss, 'r', label='Training loss (mse)')
    plt.plot(epochs, val_loss, 'g', label='Validation loss (mse)')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 2)
    plt.plot(epochs, mean_absolute_error, 'r', label='Training mae')
    plt.plot(epochs, val_mean_absolute_error, 'g', label='Validation mae')
    # plt.title('Training and validation mae')
    # plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 3)
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

def data(x, y, ddg, x_test, y_test, ddg_test, train_index, val_index):
    ## train data
    x_train = x[train_index]
    y_train = y[train_index]
    ddg_train = ddg[train_index].reshape(-1)

    ## val data
    x_val = x[val_index]
    y_val = y[val_index]
    ddg_val = ddg[val_index].reshape(-1)

    # erase rosetta energy[0:41 \union 58:end]

    # sort row default is chain, pass

    # reshape and one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    # normalization
    train_shape = x_train.shape
    test_shape = x_test.shape
    val_shape = x_val.shape
    col_train = train_shape[-1]
    col_test = test_shape[-1]
    col_val = val_shape[-1]
    x_train = x_train.reshape((-1, col_train))
    x_test = x_test.reshape((-1, col_test))
    x_val = x_val.reshape((-1,col_val))
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[np.argwhere(std == 0)] = 0.01
    x_train -= mean
    x_train /= std
    x_test -= mean
    x_test /= std
    x_val -= mean
    x_val /= std
    x_train = x_train.reshape(train_shape)
    x_test = x_test.reshape(test_shape)
    x_val = x_val.reshape(val_shape)

    # reshape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_val = x_val.reshape(x_val.shape + (1,))
    return x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val

def ieee_net(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val):
    row_num, col_num = x_train.shape[1:3]
    network = models.Sequential()
    network.add(layers.Conv2D(filters=16, kernel_size=(5, 3), activation='relu', input_shape=(row_num, col_num, 1)))
    # network.add(layers.MaxPooling2D(pool_size=(2, 2)))
    network.add(layers.Conv2D(32, (5, 3), activation='relu'))
    network.add(layers.MaxPooling2D(pool_size=(2, 2)))
    network.add(layers.Conv2D(64, (5, 3), activation='relu'))
    network.add(layers.MaxPooling2D(pool_size=(2, 2)))
    network.add(layers.Flatten())
    network.add(layers.Dense(128, activation='relu'))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dropout(0.3))
    network.add(layers.Dense(1))
    # print(network.summary())
    # rmsp = optimizers.RMSprop(lr=0.0001,  decay=0.1)
    network.compile(optimizer='rmsprop',  # SGD,adam,rmsprop
                    loss='mse',
                    metrics=['mae'])  # mae平均绝对误差（mean absolute error） accuracy
    history = network.fit(
        x_train, ddg_train, validation_data=(x_val, ddg_val),
        epochs=100, batch_size=64, verbose=0)
    history_dict = history.history
    return network, history_dict

def Conv2DMultiTaskIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val, filepth):
    K.clear_session()
    summary = False
    verbose = 1
    # setHyperParams------------------------------------------------------------------------------------------------
    batch_size = 64
    # epochs = 200
    epochs = 40

    lr = 0.0049

    optimizer = 'sgd'

    activator = 'elu'

    basic_conv2D_layers = 1
    basic_conv2D_filter_num = 16

    loop_dilation2D_layers = 2
    loop_dilation2D_filter_num = 64
    loop_dilation2D_dropout_rate = 0.2008
    dilation_lower = 2
    dilation_upper = 16

    reduce_layers = 3  # conv 3 times: 120 => 60 => 30 => 15
    # print(reduce_layers)

    reduce_conv2D_filter_num = 16
    reduce_conv2D_dropout_rate = 0.1783
    residual_stride = 2

    dense1_num = 128
    dense2_num = 32

    drop_num = 0.2605

    kernel_size = (3, 3)
    pool_size = (2, 2)
    initializer = 'random_uniform'
    padding_style = 'same'
    loss_type = 'mse'
    metrics = ('mae', pearson_r)

    my_callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.33,
            patience=5,
        ),
        callbacks.ModelCheckpoint(
            filepath=filepth,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_weights_only=True)
    ]

    if lr > 0:
        if optimizer == 'adam':
            chosed_optimizer = optimizers.Adam(lr=lr)
        elif optimizer == 'sgd':
            chosed_optimizer = optimizers.SGD(lr=lr)
        elif optimizer == 'rmsprop':
            chosed_optimizer = optimizers.RMSprop(lr=lr)

    # build --------------------------------------------------------------------------------------------------------
    ## basic Conv2D
    input_layer = Input(shape=x_train.shape[1:])
    y = layers.Conv2D(basic_conv2D_filter_num, kernel_size, padding=padding_style, kernel_initializer=initializer,
                      activation=activator)(input_layer)
    y = layers.BatchNormalization(axis=-1)(y)
    if basic_conv2D_layers == 2:
        y = layers.Conv2D(basic_conv2D_filter_num, kernel_size, padding=padding_style, kernel_initializer=initializer,
                          activation=activator)(y)
        y = layers.BatchNormalization(axis=-1)(y)

    ## loop with Conv2D with dilation (padding='same')
    for _ in range(loop_dilation2D_layers):
        y = layers.Conv2D(loop_dilation2D_filter_num, kernel_size, padding=padding_style, dilation_rate=dilation_lower,
                          kernel_initializer=initializer, activation=activator)(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(loop_dilation2D_dropout_rate)(y)
        dilation_lower *= 2
        if dilation_lower > dilation_upper:
            dilation_lower = 2

    ## Conv2D with dilation (padding='valaid') and residual block to reduce dimention.
    for _ in range(reduce_layers):
        y = layers.Conv2D(reduce_conv2D_filter_num, kernel_size, padding=padding_style, kernel_initializer=initializer,
                          activation=activator)(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(reduce_conv2D_dropout_rate)(y)
        y = layers.MaxPooling2D(pool_size, padding=padding_style)(y)
        residual = layers.Conv2D(reduce_conv2D_filter_num, 1, strides=residual_stride, padding='same')(input_layer)
        y = layers.add([y, residual])
        residual_stride *= 2

    ## flat & dense
    y = layers.Flatten()(y)
    y = layers.Dense(dense1_num, activation=activator)(y)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Dropout(drop_num)(y)
    y = layers.Dense(dense2_num, activation=activator)(y)
    y = layers.BatchNormalization(axis=-1)(y)
    y = layers.Dropout(drop_num)(y)

    output_layer = layers.Dense(1)(y)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    if summary:
        model.summary()

    model.compile(optimizer=chosed_optimizer,
                  loss=loss_type,
                  metrics=list(metrics)  # accuracy
                  )

    K.set_session(tf.Session(graph=model.output.graph))
    init = K.tf.global_variables_initializer()
    K.get_session().run(init)

    result = model.fit(x=x_train,
                       y=ddg_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       callbacks=my_callbacks,
                       validation_data=(x_val, ddg_val),
                       shuffle=True,
                       )
    # print('\n----------History:\n%s'%result.history)
    return model, result.history

if __name__ == '__main__':
    ## config TF
    CUDA_rate = '0.45'
    queueGPU(USER_MEM=5000, INTERVAL=60)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate)<0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

    modeldir = '/dl/sry/mCNN/src/Network/deepddg/regressor/model_random_train_cv_%s'%time.strftime("%Y.%m.%d.%H.%M", time.localtime())
    os.makedirs(modeldir, exist_ok=True)
    score_dict = {'pearson_coeff':[], 'std':[], 'mae':[]}
    ## load data
    data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/all/all_train_neighbor_120.npz'
    test_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/all/cro_foldall_test_center_CA_PCA_False_neighbor_120.npz'
    print('load %s'%data_pth)
    Data = np.load(data_pth)
    x, y, ddg = Data['x'], Data['y'], Data['ddg']
    test_data = np.load(test_data_pth)
    x_test, y_test, ddg_test = test_data['x'], test_data['y'], test_data['ddg']

    k_count = 1
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for train_index, val_index in skf.split(x, y):
        print('-'*50)
        print('%d-th fold is in progress.' % (k_count))
        x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val = data(x, y, ddg, x_test, y_test, ddg_test, train_index, val_index)
        print('x_train: %s'
              '\ny_train: %s'
              '\nddg_train: %s'
              '\nx_test: %s'
              '\ny_test: %s'
              '\nddg_test: %s'
              '\nx_val: %s'
              '\ny_val: %s'
              '\nddg_val: %s'
              %(x_train.shape, y_train.shape, ddg_train.shape,
                x_test.shape, y_test.shape, ddg_test.shape,
                x_val.shape, y_val.shape, ddg_val.shape))
        #
        # train & test
        #
        filepth = '%s/fold_%s_weights-improvement-{epoch:02d}-{val_loss:.4f}.h5'%(modeldir,k_count)
        # model, history_dict = ieee_net(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val)
        model, history_dict = Conv2DMultiTaskIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val, filepth)

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
        # save loss figure
        #
        try:
            figure_pth = '%s/fold_%s_lossFigure.png'%(modeldir,k_count)
            loss_plot(history_dict,outpth=figure_pth)
        except:
            print('save loss plot figure failed, fold_num: %s' % k_count)

        #
        # Load model
        #
        with open('%s/fold_%s_model.json'%(modeldir,k_count), 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = models.model_from_json(loaded_model_json)  # keras.models.model_from_yaml(yaml_string)
        loaded_model.load_weights(filepath='%s/fold_%s_weightsFinal.h5' % (modeldir,k_count))
        # loaded_model.compile(optimizer='rmsprop',  # SGD,adam,rmsprop
        #                      loss='mse',
        #                      metrics=['mae',pearson_r])  # mae平均绝对误差（mean absolute error） accuracy)

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
        k_count += 1

    #
    # save score dict
    #
    try:
        with open('%s/fold_._score.dict' % modeldir, 'w') as file:
            file.write(str(score_dict))
    except:
        print('save score dict failed')

    print('\nAVG results','-'*10)
    for key in score_dict.keys():
        print('*avg(%s): %s'%(key,np.mean(score_dict[key])))
