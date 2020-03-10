from hyperopt import Trials, STATUS_OK, tpe, rand, atpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import os
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, regularizers, optimizers
from metrics_bak import test_report,test_report_reg
from keras.utils import to_categorical


def data(neighbor_obj):
    kneighbor = neighbor_obj[:-1]
    obj_flag = neighbor_obj[-1]
    if obj_flag == 't':
        obj = 'test_report_reg'
    elif obj_flag == 'v':
        obj = 'val_mae'

    random_seed = 10
    data = np.load('E:\projects\mCNN\yanglab\mCNN-master\dataset\S2648\mCNN\wild\center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
    # data = np.load('/dl/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz/center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
    # data = np.load('/root/sry/center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
    x = data['x']
    y = data['y']
    ddg = data['ddg'].reshape(-1)

    train_num = x.shape[0]
    indices = [i for i in range(train_num)]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    positive_indices, negative_indices = ddg >= 0, ddg < 0
    x_positive, x_negative = x[positive_indices], x[negative_indices]
    y_positive, y_negative = y[positive_indices], y[negative_indices]
    ddg_positive, ddg_negative = ddg[positive_indices], ddg[negative_indices]

    left_positive, left_negative = round(0.8 * x_positive.shape[0]), round(0.8 * x_negative.shape[0])
    x_train = np.vstack((x_positive[:left_positive], x_negative[:left_negative]))
    x_test  = np.vstack((x_positive[left_positive:], x_negative[left_negative:]))
    y_train = np.vstack((y_positive[:left_positive], y_negative[:left_negative]))
    y_test  = np.vstack((y_positive[left_positive:], y_negative[left_negative:]))
    ddg_train = np.hstack((ddg_positive[:left_positive], ddg_negative[:left_negative]))
    ddg_test  = np.hstack((ddg_positive[left_positive:], ddg_negative[left_negative:]))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    class_weights_dict = dict(enumerate(class_weights))

    # sort row default is chain
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
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    return x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict,obj


def Conv2DClassifierIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict,obj):
    K.clear_session()
    CUDA = '2'
    summary = False
    verbose = 0
    # setHyperParams------------------------------------------------------------------------------------------------
    batch_size = {{choice([32,64])}}
    # batch_size = 64

    # epoch = 2
    epoch = {{choice([1,2])}}
    kernel_size=(3,3)
    pool_size=(2,2)
    initializer='random_uniform'
    padding_style='same'
    activator='relu'
    regular_rate=(0.001,0.001)

    dropout_rate = 0.1
    optimizer='adam'
    loss_type='mse'
    # metrics=('accuracy',acc, mcc, mcc_concise, recall_p, recall_n, precision_p, precision_n, tp_Concise,tn_Concise,fp_Concise,fn_Concise, recall_p_Concise,recall_n_Concise,precision_p_Concise,precision_n_Concise)
    metrics = ('mae',)
    callbacks = None
    # config TF-----------------------------------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))



    # build --------------------------------------------------------------------------------------------------------
    # mCNN feature inputs as a whole 2D array
    input_layer = Input(shape=x_train.shape[1:])
    conv1 = layers.Conv2D(16,kernel_size,kernel_initializer=initializer,activation=activator)(input_layer)
    conv2 = layers.Conv2D(32,kernel_size,kernel_initializer=initializer,activation=activator)(conv1)
    pool1 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv2)

    conv3 = layers.Conv2D(64,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool1)
    conv3_BatchNorm = layers.BatchNormalization(axis=-1)(conv3)
    pool2 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv3_BatchNorm)
    conv4 = layers.Conv2D(128,kernel_size,kernel_initializer=initializer,activation=activator,kernel_regularizer=regularizers.l1_l2(l1=regular_rate[0],l2=regular_rate[1]))(pool2)
    pool3 = layers.MaxPooling2D(pool_size,padding=padding_style)(conv4)
    flat = layers.Flatten()(pool3)

    dense = layers.Dense(1024, activation=activator)(flat)
    dense_BatchNorm = layers.BatchNormalization(axis=-1)(dense)
    drop  = layers.Dropout(dropout_rate)(dense_BatchNorm)

    # output_layer = layers.Dense(len(np.unique(y_train)),activation='softmax')(drop)
    output_layer = layers.Dense(1)(drop)
    model = models.Model(inputs=input_layer, outputs=output_layer)

    if summary:
        model.summary()

 # train(self):

    # class_weights_dict = dict(enumerate(class_weights))
    # print(class_weights_dict)

    model.compile(optimizer=optimizer,
                  loss=loss_type,
                  metrics=list(metrics) # accuracy
                  )

    K.set_session(tf.Session(graph=model.output.graph))
    init = K.tf.global_variables_initializer()
    K.get_session().run(init)

    result = model.fit(x=x_train,
                       y=ddg_train,
                       batch_size=batch_size,
                       epochs=epoch,
                       verbose=verbose,
                       callbacks=callbacks,
                       validation_data=(x_test, ddg_test),
                       shuffle=True,
                       )
    # print('\n----------History:'
    #       '\n%s'%result.history)

    if obj == 'test_report_reg':
        pearson_coeff, std = test_report_reg(model, x_test, ddg_test)
        print('\n----------Predict:\npearson_coeff: %s, std: %s'%(pearson_coeff, std))
        objective = pearson_coeff * 2 + std
        return {'loss': -objective, 'status': STATUS_OK}

    elif obj == 'val_mae':
        print(result.history)
        validation_mae = np.amax(result.history['val_mean_absolute_error'])
        print('Best validation mae of epoch:', validation_mae)
        return {'loss': validation_mae, 'status': STATUS_OK}

    # print('----------',model.evaluate(x_test, y_test))
    # score,acc,mcc_ = model.evaluate(x_test,y_test)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    # return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    # validation_acc = np.amax(result.history['val_acc'])
    # print('Best validation acc of epoch:', validation_acc)
    # return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

    # acc_test, mcc_test, recall_p_test, recall_n_test, precision_p_test, precision_n_test = test_report(
    #     model, x_test, y_test)
    # print('\n----------Predict:'
    #       '\nacc_test%s, mcc_test%s, recall_p_test%s, recall_n_test%s, precision_p_test%s, precision_n_test%s'
    #       % (acc_test, mcc_test, recall_p_test, recall_n_test, precision_p_test, precision_n_test))
    #
    # obj=acc_test+4.5*mcc_test
    # return {'loss': -obj, 'status': STATUS_OK}

if __name__ == '__main__':
    import sys
    neighbor_obj,algo_flag,max_eval,CUDA,CUDA_rate = sys.argv[1:]
    ## config TF
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if CUDA_rate != 'full':
        config = tf.ConfigProto()
        if float(CUDA_rate)<0.1:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = float(CUDA_rate)
        set_session(tf.Session(config=config))

    if algo_flag == 'tpe':
        algo = tpe.suggest
    elif algo_flag == 'rand':
        algo = rand.suggest
    elif algo_flag == 'atpe':
        algo = atpe.suggest

        best_run, best_model = optim.minimize(model=Conv2DClassifierIn1,
                                              data=data,
                                              algo=algo,
                                              eval_space=True,
                                              max_evals=int(max_eval),
                                              trials=Trials(),
                                              keep_temp=False,
                                              verbose=False,
                                              data_args=(neighbor_obj,))

        # x_train, y_train, ddg_train, x_test, y_test, ddg_test, class_weights_dict,obj = data(neighbor_obj)
        # pearson_coeff, std = test_report_reg(best_model, x_test, ddg_test)
        # print('\n----------Predict On Best Model:'
        #       '\npearson_coeff: %s, std: %s'%(pearson_coeff, std))

        print("Best performing model chosen hyper-parameters:")
        print(best_run)