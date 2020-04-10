import os, json
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras import Input, models, layers, optimizers, callbacks
from mCNN.Network.metrics import test_report_reg,pearson_r
from keras.utils import to_categorical

'This network is optimized by S2648 [0.2 vaildation|0.8 train]. suppose that we have neighbor 120'

def data(train_data_pth,test_data_pth):
    ## train data
    train_data = np.load(train_data_pth)
    x_train = train_data['x']
    y_train = train_data['y']
    ddg_train = train_data['ddg'].reshape(-1)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train.reshape(-1))
    class_weights_dict = dict(enumerate(class_weights))
    ## valid data
    random_seed = 10
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

    left_positive, left_negative = round(0.8 * x_positive.shape[0]), round(0.8 * x_negative.shape[0])
    x_train = np.vstack((x_positive[:left_positive], x_negative[:left_negative]))
    x_val = np.vstack((x_positive[left_positive:], x_negative[left_negative:]))
    y_train = np.vstack((y_positive[:left_positive], y_negative[:left_negative]))
    y_val = np.vstack((y_positive[left_positive:], y_negative[left_negative:]))
    ddg_train = np.hstack((ddg_positive[:left_positive], ddg_negative[:left_negative]))
    ddg_val = np.hstack((ddg_positive[left_positive:], ddg_negative[left_negative:]))

    ## test data
    test_data = np.load(test_data_pth)
    x_test = test_data['x']
    y_test = test_data['y']
    ddg_test = test_data['ddg'].reshape(-1)

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
    return x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val, class_weights_dict


def Conv2DMultiTaskIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val, class_weights_dict,modeldir):
    K.clear_session()
    summary = False
    verbose = 0
    # setHyperParams------------------------------------------------------------------------------------------------
    batch_size = 64
    epochs = 200

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
            factor=0.5,
            patience=5,
        )
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
                       validation_data=(x_test, ddg_test),
                       shuffle=True,
                       )
    # print('\n----------History:\n%s'%result.history)
    model_json = model.to_json()
    with open(modeldir+'/model.json', 'w') as json_file:
        json_file.write(model_json)
    return model,result.history

if __name__ == '__main__':
    CUDA,CUDA_rate = '3','full'
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
    score_dict = {'pearson_coeff':[], 'std':[], 'mae':[]}
    data_basedir = '/dl/sry/mCNN/dataset/S2648/npz/wild/cross_valid'

    for i in range(5):
        fold_num = i+1
        print('--cross validation begin, fold %s is processing.'%fold_num)
        train_data_pth = data_basedir+'/pos_fold%s_train_center_CA_PCA_False_neighbor_120.npz'%fold_num
        test_data_pth = data_basedir+'/pos_fold%s_test_center_CA_PCA_False_neighbor_120.npz'%fold_num
        modeldir = '/dl/sry/mCNN/dataset/S2648/saved_model/wild/cross_valid/regressor_fold%s'%fold_num
        os.makedirs(modeldir,exist_ok=True)
        x_train, y_train, ddg_train, x_test, y_test, ddg_test, x_val, y_val, ddg_val, class_weights_dict = data(train_data_pth,test_data_pth)

        model,history_dict = Conv2DMultiTaskIn1(x_train, y_train, ddg_train, x_test, y_test, ddg_test,x_val,y_val,ddg_val, class_weights_dict,modeldir)
        try:
            model.save_weights(filepath='%s/weights.hdf5'%modeldir)
        except:
            print('save final model weights failed, fold_num: %s'%fold_num)
        try:
            with open('%s/history.json'%modeldir, 'w') as file:
                file.write(json.dumps(history_dict))
        except:
            print('save history_dict to json failed, fold_num: %s' % fold_num)

        pearson_coeff, std, mae = test_report_reg(model, x_test, ddg_test)
        print('\n----------Predict:'
              '\npearson_coeff: %s, std: %s, mae: %s'
              % (pearson_coeff, std, mae))
        score_dict['pearson_coeff'].append(pearson_coeff)
        score_dict['std'].append(std)
        score_dict['mae'].append(mae)
    #print average
    with open('/dl/sry/mCNN/dataset/S2648/saved_model/wild/cross_valid/regressor_avg_scores.json', 'w') as file:
        file.write(json.dumps(score_dict))
    print('\nAVG results','-'*10)
    for key in score_dict.keys():
        print('*avg(%s): %s'%(key,np.mean(score_dict[key])))


