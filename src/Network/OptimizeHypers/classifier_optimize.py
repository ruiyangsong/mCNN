#coding=utf-8

try:
    from hyperopt import Trials, STATUS_OK, tpe, rand, atpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, loguniform
except:
    pass

try:
    import os
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    from sklearn.utils import class_weight
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    from keras import backend as K
except:
    pass

try:
    from keras.backend.tensorflow_backend import set_session
except:
    pass

try:
    from keras import Input, models, layers, regularizers, optimizers, callbacks
except:
    pass

try:
    from metrics_bak import test_report
except:
    pass

try:
    from keras.utils import to_categorical
except:
    pass

try:
    import sys
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
kneighbor = '120'
random_seed = 10
# data = np.load('E:\projects\mCNN\yanglab\mCNN-master\dataset\S2648\mCNN\wild\center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
data = np.load('/dl/sry/mCNN/dataset/S2648/feature/mCNN/wild/npz/center_CA_PCA_False_neighbor_%s.npz'%kneighbor)
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


def keras_fmin_fnct(space):

        summary = True
        verbose = 0
        # setHyperParams------------------------------------------------------------------------------------------------
        batch_size = space['batch_size']
        epochs = space['epochs']

        lr = space['lr']

        optimizer = space['optimizer']

        activator = space['activator']

        basic_conv2D_layers     = space['basic_conv2D_layers']
        basic_conv2D_filter_num = space['basic_conv2D_filter_num']

        loop_dilation2D_layers = space['loop_dilation2D_layers']
        loop_dilation2D_filter_num = space['loop_dilation2D_filter_num']#used in the loop
        loop_dilation2D_dropout_rate = space['loop_dilation2D_dropout_rate']
        dilation_lower = 2
        dilation_upper = 16

        reduce_layers = 3  # conv 3 times: 120 => 60 => 30 => 15
        reduce_conv2D_filter_num = space['reduce_conv2D_filter_num']#used for reduce dimention
        reduce_conv2D_dropout_rate = space['reduce_conv2D_dropout_rate']
        residual_stride = 2

        dense1_num = space['dense1_num']
        dense2_num = space['dense2_num']

        drop1_num = space['drop1_num']
        drop2_num = space['drop2_num']

        kernel_size=(3,3)
        pool_size=(2,2)
        initializer='random_uniform'
        padding_style='same'
        loss_type='binary_crossentropy'
        metrics = ('accuracy',)


        my_callbacks = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                )
            ]

        if lr > 0:
            if optimizer == 'adam':
                chosed_optimizer = optimizers.Adam(lr=lr)
            elif optimizer == 'sgd':
                chosed_optimizer = optimizers.SGD(lr=lr)
            elif optimizer == 'rmsprop':
                chosed_optimizer = optimizers.RMSprop(lr=lr)
            elif optimizer == 'adagrad':
                chosed_optimizer = optimizers.Adagrad(lr=lr)
            elif optimizer == 'adadelta':
                chosed_optimizer = optimizers.Adadelta(lr=lr)
            elif optimizer == 'adamax':
                chosed_optimizer = optimizers.Adamax(lr=lr)
            elif optimizer == 'nadam':
                chosed_optimizer = optimizers.Nadam(lr=lr)

        # build --------------------------------------------------------------------------------------------------------
        ## basic Conv2D
        input_layer = Input(shape=x_train.shape[1:])
        y = layers.Conv2D(basic_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(input_layer)
        y = layers.BatchNormalization(axis=-1)(y)
        if basic_conv2D_layers == 2:
            y = layers.Conv2D(basic_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)

        ## loop with Conv2D with dilation (padding='same')
        for _ in range(loop_dilation2D_layers):
            y = layers.Conv2D(loop_dilation2D_filter_num, kernel_size, padding=padding_style,dilation_rate=dilation_lower, kernel_initializer=initializer, activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.Dropout(loop_dilation2D_dropout_rate)(y)
            dilation_lower*=2
            if dilation_lower>dilation_upper:
                dilation_lower=2

        ## Conv2D with dilation (padding='valaid') and residual block to reduce dimention.
        for _ in range(reduce_layers):
            y = layers.Conv2D(reduce_conv2D_filter_num,kernel_size,padding=padding_style,kernel_initializer=initializer,activation=activator)(y)
            y = layers.BatchNormalization(axis=-1)(y)
            y = layers.Dropout(reduce_conv2D_dropout_rate)(y)
            y = layers.MaxPooling2D(pool_size,padding=padding_style)(y)
            residual = layers.Conv2D(reduce_conv2D_filter_num, 1, strides=residual_stride, padding='same')(input_layer)
            y = layers.add([y, residual])
            residual_stride*=2

        ## flat & dense
        y = layers.Flatten()(y)
        y = layers.Dense(dense1_num, activation=activator)(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y  = layers.Dropout(drop1_num)(y)
        y = layers.Dense(dense2_num, activation=activator)(y)
        y = layers.BatchNormalization(axis=-1)(y)
        y = layers.Dropout(drop2_num)(y)

        output_layer = layers.Dense(len(np.unique(y_train)),activation='softmax')(y)

        model = models.Model(inputs=input_layer, outputs=output_layer)

        if summary:
            model.summary()


        model.compile(optimizer=chosed_optimizer,
                      loss=loss_type,
                      metrics=list(metrics) # accuracy
                      )

        K.set_session(tf.Session(graph=model.output.graph))
        init = K.tf.global_variables_initializer()
        K.get_session().run(init)

        result = model.fit(x=x_train,
                           y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose,
                  callbacks=my_callbacks,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  class_weight=class_weights_dict
                  )
        print('\n----------History:'
              '\n%s'%result.history)
        acc_test, mcc_test, recall_p_test, recall_n_test, precision_p_test, precision_n_test = test_report(
            model, x_test, y_test)
        print('\n----------Predict:'
              '\nacc_test: %s, mcc_test: %s, recall_p_test: %s, recall_n_test: %s, precision_p_test: %s, precision_n_test: %s'
              % (acc_test, mcc_test, recall_p_test, recall_n_test, precision_p_test, precision_n_test))
        # return model

        # validation_acc = np.amax(result.history['val_acc'])
        # print('Best validation acc of epoch:', validation_acc)
        # return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

        obj = acc_test + 5 * mcc_test + recall_p_test+recall_n_test+precision_p_test+precision_n_test
        return {'loss': -obj, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'batch_size': hp.choice('batch_size', [32,64,128,256]),
        'epochs': hp.choice('epochs', [25,50,75,100,125,150,175,200]),
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'optimizer': hp.choice('optimizer', ['adam','sgd','rmsprop','adagrad','adadelta','adamax','nadam']),
        'activator': hp.choice('activator', ['elu', 'relu', 'tanh']),
        'basic_conv2D_layers': hp.choice('basic_conv2D_layers', [1, 2]),
        'basic_conv2D_filter_num': hp.choice('basic_conv2D_filter_num', [16, 32]),
        'loop_dilation2D_layers': hp.choice('loop_dilation2D_layers', [1, 2, 3, 4, 5]),
        'loop_dilation2D_filter_num': hp.choice('loop_dilation2D_filter_num', [16, 32, 64]),
        'loop_dilation2D_dropout_rate': hp.uniform('loop_dilation2D_dropout_rate', 0.001, 0.35),
        'reduce_conv2D_filter_num': hp.choice('reduce_conv2D_filter_num', [8, 16, 32]),
        'reduce_conv2D_dropout_rate': hp.uniform('reduce_conv2D_dropout_rate', 0.001, 0.25),
        'dense1_num': hp.choice('dense1_num', [128, 256, 512]),
        'dense2_num': hp.choice('dense2_num', [32, 64, 128]),
        'drop1_num': hp.uniform('drop1_num', 0.0001, 0.3),
        'drop2_num': hp.uniform('drop2_num', 0.0001, 0.1),
    }
