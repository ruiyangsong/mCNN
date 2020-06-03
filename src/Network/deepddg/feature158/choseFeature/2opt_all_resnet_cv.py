from hyperopt import Trials, STATUS_OK, tpe, rand, atpe, hp
from hyperas import optim
from hyperas.distributions import choice, uniform, loguniform

import os, json, time
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks, regularizers,initializers
from mCNN.Network.metrics import test_report_reg, pearson_r, rmse
from mCNN.Network.lossFunction import logcosh
from keras.utils import to_categorical
from save_train import save_train_cv

def data():
    return 0

def Conv1DRegressorIn1(flag):
    K.clear_session()
    current_neighbor              = space['neighbor']
    current_idx_idx               = space['idx_idx']

    current_dense_num             = space['dense_num']
    current_dilation1D_layers     = space['dilation1D_layers']
    current_dilation1D_filter_num = space['dilation1D_filter_num']
    current_reduce1D_filter_num   = space['reduce1D_filter_num']

    summary = True
    verbose = 0

    #
    # setHyperParams
    #
    ## hypers for data
    neighbor = {{choice([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])}}
    idx_idx = {{choice([0,1,2,3,4,5,6,7,8])}}
    idx_lst = [
        [x for x in range(158) if x not in [24, 26]],  # 去除无用特征
        [x for x in range(158) if x not in [24, 26] + [x for x in range(1, 6)] + [x for x in range(16, 22)] + [40, 42]], # 去除无用特征+冗余特征
        [x for x in range(158) if x not in [24, 26] + [x for x in range(0, 22)]],  # 去除无用特征+方位特征
        [x for x in range(158) if x not in [24, 26] + [22, 23, 26, 37, 38]],  # 去除无用特征+深度特征
        [x for x in range(158) if x not in [24, 26] + [x for x in range(27, 37)] + [x for x in range(40, 46)]],# 去除无用特征+二级结构信息
        # [x for x in range(158) if x not in [24, 26] + [x for x in range(27, 34)] + [x for x in range(40, 46)]],# 去除无用特征+二级结构信息1
        # [x for x in range(158) if x not in [24, 26] + [x for x in range(34, 37)] + [x for x in range(40, 46)]],# 去除无用特征+二级结构信息2
        [x for x in range(158) if x not in [24, 26] + [46, 47]],  # 去除无用特征+实验条件
        [x for x in range(158) if x not in [24, 26] + [39] + [x for x in range(57, 61)] + [x for x in range(48, 57)] + [x for x in range(61, 81)] + [x for x in range(140, 155)]], # 去除无用特征+所有原子编码
        # [x for x in range(158) if x not in [24, 26] + [39] + [x for x in range(57, 61)] + [x for x in range(48, 57)] + [x for x in range(140, 145)]],# 去除无用特征+原子编码1
        # [x for x in range(158) if x not in [24, 26] + [39] + [x for x in range(57, 61)] + [x for x in range(61, 77)] + [x for x in range(145, 153)]],# 去除无用特征+原子编码2
        # [x for x in range(158) if x not in [24, 26] + [39] + [x for x in range(57, 61)] + [x for x in range(77, 81)] + [x for x in range(153, 155)]],# 去除无用特征+原子编码3
        [x for x in range(158) if x not in [24, 26] + [x for x in range(81, 98)]],  # 去除无用特征+rosetta_energy
        [x for x in range(158) if x not in [24, 26] + [x for x in range(98, 140)] + [x for x in range(155, 158)]]# 去除无用特征+msa
    ]
    idx = idx_lst[idx_idx]
    ## hypers for net
    lr = 1e-4  # 0.0001
    batch_size = 32
    epochs = 200
    padding_style = 'same'
    activator_Conv1D = 'elu'
    activator_Dense = 'tanh'
    dense_num = {{choice([64, 96, 128])}}# 64
    dilation1D_layers = {{choice([8, 16, 32])}}#8
    dilation1D_filter_num = {{choice([16, 32])}}#16
    dilation_lower = 1
    dilation_upper = 16
    dropout_rate_dilation = 0.25
    dropout_rate_reduce = 0.25
    dropout_rate_dense = 0.25
    initializer_Conv1D = initializers.lecun_uniform(seed=527)
    initializer_Dense = initializers.he_normal(seed=527)
    kernel_size = 5
    l2_rate = 0.001
    loss_type = logcosh
    metrics = ('mae', pearson_r, rmse)
    pool_size = 2
    reduce_layers = 5 # 110 -> 55 -> 28 -> 14 -> 7 -> 4, 50 - 25 - 13 - 7 - 4 - 2
    reduce1D_filter_num = {{choice([16, 32, 64])}}#32
    residual_stride = 2

    def _data(fold_num, neighbor, idx):
        train_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_fold%s_train_center_CA_PCA_False_neighbor_140.npz' % fold_num
        val_data_pth = '/dl/sry/mCNN/dataset/deepddg/npz/wild/cross_valid/cro_fold%s_valid_center_CA_PCA_False_neighbor_140.npz' % fold_num

        ## train data
        train_data = np.load(train_data_pth)
        x_train = train_data['x']
        y_train = train_data['y']
        ddg_train = train_data['ddg'].reshape(-1)
        ## select kneighbor atoms
        x_train_kneighbor_lst = []
        for sample in x_train:
            dist_arr = sample[:, 0]
            indices = sorted(dist_arr.argsort()[:neighbor])
            x_train_kneighbor_lst.append(sample[indices, :])
        x_train = np.array(x_train_kneighbor_lst)
        ## idx
        x_train = x_train[:, :, idx]

        ## val data
        val_data = np.load(val_data_pth)
        x_val = val_data['x']
        y_val = val_data['y']
        ddg_val = val_data['ddg'].reshape(-1)
        ## select kneighbor atoms
        x_val_kneighbor_lst = []
        for sample in x_val:
            dist_arr = sample[:, 0]
            indices = sorted(dist_arr.argsort()[:neighbor])
            x_val_kneighbor_lst.append(sample[indices, :])
        x_val = np.array(x_val_kneighbor_lst)
        ##  idx
        x_val = x_val[:, :, idx]

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
        print('x_train: %s'
              '\ny_train: %s'
              '\nddg_train: %s'
              '\nx_val: %s'
              '\ny_val: %s'
              '\nddg_val: %s'
              % (x_train.shape, y_train.shape, ddg_train.shape,
                 x_val.shape, y_val.shape, ddg_val.shape))
        return x_train, y_train, ddg_train, x_val, y_val, ddg_val

    #
    # cross_valid
    #
    hyper_param_tag = '%s_%s_%s_%s_%s_%s' % (
        current_neighbor, current_idx_idx, current_dense_num,
        current_dilation1D_layers, current_dilation1D_filter_num, current_reduce1D_filter_num)
    modeldir = '/dl/sry/projects/from_hp/mCNN/src/Network/deepddg/opt_all_resnet/model/%s-%s' % (
        hyper_param_tag, time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime()))
    os.makedirs(modeldir, exist_ok=True)
    opt_lst = []

    for k_count in range(1,11):
        print('\n** fold %s is processing **\n'%k_count)
        filepth = '%s/fold_%s_weights-best.h5' % (modeldir, k_count)
        my_callbacks = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.33,
                patience=5,
                verbose=verbose,
                mode='min',
                min_lr=1e-8,
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=verbose
            ),
            callbacks.ModelCheckpoint(
                filepath=filepth,
                monitor='val_mean_absolute_error',
                verbose=verbose,
                save_best_only=True,
                mode='min',
                save_weights_only=True)
        ]

        x_train, y_train, ddg_train, x_val, y_val, ddg_val = _data(k_count,neighbor,idx)
        row_num, col_num = x_train.shape[1:3]
        #
        # build net
        #
        input_layer = Input(shape=(row_num, col_num),name='input_layer')
        y = layers.Conv1D(
            filters=dilation1D_filter_num,
            kernel_size=1,
            padding=padding_style,
            activation=activator_Conv1D,
            kernel_initializer=initializer_Conv1D,
            kernel_regularizer=regularizers.l2(l2_rate),
            name='first_conv_layer',
            trainable=True
        )(input_layer)
        res = layers.BatchNormalization(axis=-1,name='first_BN_layer')(y)

        ## loop with Conv1D with dilation (padding='same')
        for _ in range(dilation1D_layers):
            y = layers.SeparableConv1D(
                filters=dilation1D_filter_num,
                kernel_size=kernel_size,
                padding=padding_style,
                dilation_rate=dilation_lower,
                activation=activator_Conv1D,
                depthwise_initializer=initializer_Conv1D,
                pointwise_initializer=initializer_Conv1D,
                depthwise_regularizer=regularizers.l2(l2_rate),
                pointwise_regularizer=regularizers.l2(l2_rate)
            )(res)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.Dropout(dropout_rate_dilation)(y)

            y = layers.SeparableConv1D(
                filters=dilation1D_filter_num,
                kernel_size=kernel_size,
                padding=padding_style,
                dilation_rate=dilation_lower,
                activation=activator_Conv1D,
                depthwise_initializer=initializer_Conv1D,
                pointwise_initializer=initializer_Conv1D,
                depthwise_regularizer=regularizers.l2(l2_rate),
                pointwise_regularizer=regularizers.l2(l2_rate)
            )(y)
            y = layers.BatchNormalization(axis=-1)(y)

            res = layers.add([y, res])

            dilation_lower *= 2
            if dilation_lower > dilation_upper:
                dilation_lower = 1

        ## Conv1D with dilation (padding='valaid') and residual block to reduce dimention.
        for _ in range(reduce_layers):
            y = layers.SeparableConv1D(
                filters=reduce1D_filter_num,
                kernel_size=kernel_size,
                padding=padding_style,
                activation=activator_Conv1D,
                depthwise_initializer=initializer_Conv1D,
                pointwise_initializer=initializer_Conv1D,
                depthwise_regularizer=regularizers.l2(l2_rate),
                pointwise_regularizer=regularizers.l2(l2_rate)
            )(res)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.Dropout(dropout_rate_reduce)(y)
            y = layers.MaxPooling1D(pool_size, padding=padding_style)(y)
            res = layers.SeparableConv1D(
                filters=reduce1D_filter_num,
                kernel_size=kernel_size,
                strides=residual_stride,
                padding=padding_style,
                activation=activator_Conv1D,
                depthwise_initializer=initializer_Conv1D,
                pointwise_initializer=initializer_Conv1D,
                depthwise_regularizer=regularizers.l2(l2_rate),
                pointwise_regularizer=regularizers.l2(l2_rate)
            )(res)
            res = layers.add([y, res])

        ## flat & dense
        y = layers.Flatten()(y)
        y = layers.Dense(dense_num, activation=activator_Dense, kernel_initializer=initializer_Dense)(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(dropout_rate_dense)(y)

        output_layer = layers.Dense(1, activation='linear')(y)

        model = models.Model(inputs=input_layer, outputs=output_layer)
        # print(model.get_layer(index=2).trainable)
        # print(model.get_layer(name='first_conv').trainable)
        if summary:
            trainable_count = int(
                np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
            non_trainable_count = int(
                np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

            print('Total params: {:,}'.format(trainable_count + non_trainable_count))
            print('Trainable params: {:,}'.format(trainable_count))
            print('Non-trainable params: {:,}'.format(non_trainable_count))
            # model.summary()

        adam = optimizers.Adam(lr=lr)
        model.compile(optimizer=adam,  # 'rmsprop',  # SGD,adam,rmsprop
                      loss=loss_type,
                      metrics=list(metrics))  # mae平均绝对误差（mean absolute error） accuracy
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
        #
        # save
        #
        save_train_cv(model, modeldir, result.history,k_count)
        opt_lst.append(np.mean(result.history['val_mean_absolute_error'][-10:]))
    opt_loss = np.mean(opt_lst)
    #
    # print hyper combination group and current loss value
    #
    print('\n@current_hyper_tag: %s'
          '\n@current optmized_loss: %s'
          %(hyper_param_tag, opt_loss))
    # return {'loss': validation_loss, 'status': STATUS_OK, 'model':model}
    return {'loss': opt_loss, 'status': STATUS_OK}

def config_tf():
    from mCNN.queueGPU import queueGPU
    CUDA_rate = '0.25'
    ## config TF
    queueGPU(USER_MEM=3500, INTERVAL=60)
    # os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate) < 0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

if __name__ == '__main__':
    import sys
    if len(sys.argv)==1:
        print('Usage: %s [max_eval]'%sys.argv[0])
        sys.exit(0)

    config_tf()

    algo_flag = 'tpe'
    max_eval  = sys.argv[1]
    if algo_flag == 'tpe':
        algo = tpe.suggest
    elif algo_flag == 'rand':
        algo = rand.suggest
    elif algo_flag == 'atpe':
        algo = atpe.suggest
    best_run, best_model, search_space = optim.minimize(
        model=Conv1DRegressorIn1,
        data=data,
        algo=algo,
        max_evals=int(max_eval),
        trials=Trials(),#A hyperopt trials object, used to store intermediate results for all optimization runs
        functions=None,
        rseed=527, #Integer random seed for experiments
        notebook_name=None,
        verbose=True,
        eval_space=True, #Evaluate the best run in the search space such that 'choice's contain actually meaningful values instead of mere indices
        return_space=True, #Return the hyperopt search space object (e.g. for further processing) as last return value
        keep_temp=False,
        data_args=None)

    # print("The hyperopt search space:"
    #       "\n%s" % search_space)
    print("Best performing model chosen hyper-parameters:"
          "\n%s" % best_run)
