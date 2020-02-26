from hyperopt import Trials, STATUS_OK, tpe
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
from metrics import pearson_r, rmse, mcc, recall, recall_p, recall_n, precision, precision_p, precision_n
from CallBack import TrainCallback
from keras.utils import to_categorical


def data():
    random_seed = 10
    data = np.load('E:\projects\mCNN\yanglab\mCNN-master\dataset\S2648\mCNN\wild\center_CA_PCA_False_neighbor_30.npz')
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
    left_positive, left_negative = round(0.8 * x_positive.shape[0]), round(0.8 * x_negative.shape[0])
    x_train, x_test = np.vstack((x_positive[:left_positive], x_negative[:left_negative])), np.vstack(
        (x_positive[left_positive:], x_negative[left_negative:]))
    y_train, y_test = np.vstack((y_positive[:left_positive], y_negative[:left_negative])), np.vstack(
        (y_positive[left_positive:], y_negative[left_negative:]))
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
    return x_train, y_train, x_test, y_test


def Conv2DClassifierIn1(x_train,y_train,x_test,y_test):
        CUDA = '2'
        summary = False
        verbose = 1
        # setHyperParams------------------------------------------------------------------------------------------------
        batch_size = 128
        # batch_size = 16
        # epoch = {{choice([2, 4, 6])}}
        epoch = 25
        kernel_size=(3,3)
        pool_size=(2,2)
        initializer='random_uniform'
        padding_style='same'
        activator='relu'
        regular_rate=(0.001,0.001)
        # dropout_rate = {{uniform(0, 1)}}
        dropout_rate = 0.1
        optimizer='adam'
        loss_type='binary_crossentropy'
        metrics=('accuracy', mcc, recall, recall_p, recall_n, precision, precision_p, precision_n)
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

        output_layer = layers.Dense(len(np.unique(y_train)),activation='softmax')(drop)
        model = models.Model(inputs=input_layer, outputs=output_layer)

        if summary:
            model.summary()

 # train(self):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
        class_weights_dict = dict(enumerate(class_weights))
        model.compile(optimizer=optimizer,
                      loss=loss_type,
                      metrics=list(metrics) # accuracy
                      )

        result = model.fit(x=x_train,
                           y=y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  verbose=verbose,
                  callbacks=callbacks,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  class_weight=class_weights_dict
                  )
        validation_acc = np.amax(result.history['val_acc'])
        print('Best validation acc of epoch:', validation_acc)
        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data()
    Conv2DClassifierIn1(x_train, y_train, x_test, y_test)

    #
    # best_run, best_model = optim.minimize(model=Conv2DClassifierIn1,
    #                                       data=data,
    #                                       algo=tpe.suggest,
    #                                       max_evals=25,
    #                                       trials=Trials())
    # X_train, Y_train, X_test, Y_test = data()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(X_test, Y_test))
    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)