# Keras踩坑

## 0.自定义metrics
**!! ATTENTION !!**
* For those custom metrics, the average accross minibatches is namely not equal to the metric evaluated on the whole dataset.
* The metric on the validation set is calculated in batches, and then averaged (of course the trained model at the end of the epoch is used,
 in contrast to how the metric score is calculated for the training set)
* 1. How to compute precision and recall in Keras? --> https://www.thinbug.com/q/43076609
* 2. How are metrics computed in Keras? --> https://stackoverflow.com/questions/49359489/how-are-metrics-computed-in-keras

## 1.设置earlystopping

```python
filepath = model_snapshot_directory + '/' + 'lstm_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(X_train,y_train,epochs=100,batch_size=128,
          verbose=1,callbacks=[checkpoint],validation_data=(X_test,y_test)
```

**checkpoint设置的监控值是monitor=val_loss,当val_loss值不发生很大的改善就不保存模型.**

## 2.使用hyperas

```python
best_run,best_model=optim.minimize(model=train,data=prepare_data,algo=tpe.suggest, max_evals=100,trials=Trials())
```

在这里max_eval=100表示在训练过程中要对不同的组合评估100次，每一次的模型参数都不一样。</span>这个只可以根据实际参数的多少来设置，越大可能训练的模型就越多。</p>

## 3.模型的评估

```python
best_model.evaluate(X_test,y_test)
```

**这个evaluate 的返回值是一个元组（score，acc），loss值=-score**

## 4.model.fit的返回值

```python
hist=model.fit(X_train, y_train, epochs=100, batch_size={{choice([64, 128, 256])}}, verbose=1,
                             callbacks=callback_list, validation_data=(X_test, y_test))
h1=hist.history
acc_=np.asarray(h1['acc'])
loss_=np.asarray((h1['loss']))
val_acc=np.asarray(h1['val_acc'])
val_loss=np.asarray(h1['val_loss'])
acc_and_loss=np.column_stack((acc_,loss_,val_acc,val_loss))
save_file_mlp = model_snapshot_directory+'/mlp_run_' + '_' + str(globalvars.globalVar) + '.txt'
with open(save_file_mlp, 'w') as f:
    np.savetxt(save_file_mlp, acc_and_loss, delimiter=" ")
```


**fit()函数返回一个名为history的变量，其中包含损失追踪以及在编译模型时指定的任何其他指标，这些分数都记录在每个训练轮数的末尾。**

**可以使用Matplotlib库绘制模型的性能图:**

```python
from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
```


## 5.诊断LSTM网络模型的过拟合和欠拟合

<p><a href="https://baijiahao.baidu.com/s?id=1577431637601070077&amp;wfr=spider&amp;for=pc" rel="nofollow">https://baijiahao.baidu.com/s?id=1577431637601070077&amp;wfr=spider&amp;for=pc</a></p>

## 6.使用多GPU跑模型，并保存模型

**新建一个py文件，内容如下：**

```python
from keras.callbacks import ModelCheckpoint
class AltModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, alternate_model, **kwargs):
        """
        Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
        :param filepath:
        :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                                gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                                "template model" to be saved each checkpoint.
        :param kwargs:          Passed to ModelCheckpoint.
        """

        self.alternate_model = alternate_model
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before
```

**然后在训练的文件中：**

```python
from alt_model_checkpoint import AltModelCheckpoint
from keras.models import Model
from keras.utils import multi_gpu_model
base_model = Model(...)
gpu_model = multi_gpu_model(base_model，numbers_of_gpu)
gpu_model.compile(...)
gpu_model.fit(..., callbacks=[
    AltModelCheckpoint('save/path/for/model.hdf5', base_model)
])
```

**如果要<span style="color:#ff4635;">加上earlystopping</span>，则修改fit 的内容，比如：**

```python
hist = gpu_model.fit(X_train, y_train,
                     batch_size={{choice([64, 128, 256])}},
                     epochs=100,
                     verbose=1, 
                     callbacks=[
                         AltModelCheckpoint(
                             filepath,
                             model,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min'
                         )
                     ],
                     validation_data=(X_test, y_test))
```

**<span style="color:#f33b45;">因为AltModelCheckpoint是继承自ModelCheckpoint，所以可以直接添加。</span>**

**这个是使用多GPU的例子**

1. 主要调用了multi_gpu_model这个函数
2. 在训练的时候，保存检查点模型，自定义了一个函数，保存的是base_model而不是gpu_model。
在模型保存之后，load的时候：
* 用load_model load保存的模型文件
* 需要gpu_model=multi_gpu_model(base_model)
gpu_model.complie()

3.做预测: gpu_model.predict()

<p><a href="https://github.com/keras-team/keras/issues/9342">https://github.com/keras-team/keras/issues/9342</a></p>

**在keras的 saving.py文件中,添加这个：**

```python
# ... earlier get_json_type code
# NOTE: Hacky fix to serialize Dimension objects.
from tensorflow.python.framework.tensor_shape import Dimension
if type(obj) == Dimension:
  return int(obj)
# original error raised here
raise TypeError('Not JSON Serializable:', obj)
```

**Keras调用多GPU的例子： <a href="https://www.jianshu.com/p/d57595dac5a9" rel="nofollow">https://www.jianshu.com/p/d57595dac5a9</a>**

**<span style="color:#ff4635;">多GPU+earlystopping+hyperas进行调参</span>**

```python
# in the function train
def train():
        ...
        # first distribute GPUs according to the gpu which you possess
    gpu_model = multi_gpu_model(model, gpus=2)
    gpu_model.compile(loss=loss_fn, optimizer=optim, metrics=['accuracy'])
    # set earlystopping using ModelCheckPoint
    filepath='...'
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=True, mode='min')
    callback_list = [early_stopping, checkpointer]
    # train the model
    hist = gpu_model.fit(X_train, y_train, epochs=100, batch_size=128,
                         verbose=1, callbacks=callback_list, validation_data=(X_test, y_test))
    gpu_model.load_weights(filepath)
    score, acc = gpu_model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': gpu_model}
```

**保存的时候使用gpu_model,并且在return的时候也是gpu_model**

**注意：使用多GPU训练的模型需要依旧使用多GPU来做预测，比如在节点g-1-4上训练，就仍要使用g-1-4predict，并且使用相同的GPU数量。**

**<span style="color:#ff0000;"><strong>注</strong></span>：</p>

<p>1.在使用多GPU并且设置ModelCheckpoint的时候，不能设置save_model_only=True,只有设置save_weights_only=True的时候，才能够正常训练模型，并且代码运行正常结束。</p>

<p>2.使用多GPU+Hyperas+ModelCheckpoint</strong>时，因为按照第一条只能保存权重，而且hyperas的优化得到的best_model不能进行正常报错，会报错，can not pickle the module。并且即使能够保存，在进行预测的时候也要重构模型，并且要使用训练时相同的GPU数，但是却无法得知最优的是哪一个权重。因此就无法进行预测。</p>

<p>解决方法是：</p>

```python
def train():
        ...
        # first distribute GPUs according to the gpu which you possess
    gpu_model = multi_gpu_model(model, gpus=2)
    gpu_model.compile(loss=loss_fn, optimizer=optim, metrics=['accuracy'])
    # set earlystopping using ModelCheckPoint
    filepath='...'
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=True, mode='min')
    callback_list = [early_stopping, checkpointer]
    # train the model
    hist = gpu_model.fit(X_train, y_train, epochs=100, batch_size=128,
                         verbose=1, callbacks=callback_list, validation_data=(X_test, y_test))
    score, acc = gpu_model.evaluate(X_test, y_test, verbose=0)
    model.save(filepath)    # 这一段一定要放在gpu_model.evaluate下面,否则会出错
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
```

<p>注意保存的时候是model，返回的模型是model</p>

<p>虽然没有compile model但是在fit结束后，model的权重就是gpu_model的权重。   </p>

<p> 经过测试，是可以得到best_model的。并且可以在单GPU上进行预测。</p>

## 7.keras模型的多输入（此处使用了多GPU以及hyperas调参工具）

```python
input_embed = Input(shape=(700,), name='input_embed')
input_extra = Input(shape=(700, 25,), name='input_extra')
embedded = Embedding(num_amino_acids, 50, input_length=700)(input_embed)
x = concatenate([embedded, input_extra], axis=2)
......
x = BatchNormalization()(x)
output = Activation(activation='sigmoid')(x)
model = Model(inputs=[input_embed, input_extra], outputs=output)

gpu_model = multi_gpu_model(model, 4)
gpu_model.compile(...)
callback_list = [early_stopping, checkpointer]
hist = gpu_model.fit(x={'input_embed': X_all, 'input_extra': X_extra},
                     y=y_all,
                     epochs=100,
                     batch_size=256,
                     verbose=1,
                     callbacks=callback_list,
                     class_weight=class_weights,
                     validation_split=0.2)
```
虽然设置了validation_split但是在训练的时候只会在每一个epoch验证，每一个batch没有验证，而且，对于多输入来说，不能对validation_data里面添加多输入.

<p>注意：此处多次踩坑，特别重要，下面的代码中的X_train,X_extra,y_train很重要，必须得在prepare_data（）函数中是这个名称,否则会报错名称不存在</p>

```python
def prepare_data():
    ......
    return (X_train,X_extra,y_train)

if __name__ == "__main__":
    best_run,best_model=optim.minimize(model=train,data=prepare_data,algo=tpe.suggest,
                                       max_evals=30,trials=Trials())
    X_train,X_extra,y_train=prepare_data()
```

## 8.设置earlystoppping的monitor监控自定义的值，比如auc值。

<p>方法一：经过试验，当使用hyperas进行调参时，会报错，说没有auc_roc</p>

```python
from tensorflow.contrib.metrics import streaming_auc
def auc_roc(y_true,y_pred):
    value,update_op=streaming_auc(y_pred,y_true)
    # find all variables created for this metric
    metric_vars=[i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def train():
        ......
        gpu_model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy',auc_roc])
    early_stopping = EarlyStopping(monitor='val_auc_roc', patience=20, mode='max')
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_auc_roc', verbose=1,
                               save_best_only=True,save_weights_only=True, mode='max')
```

## 9.keras+hyperas调参时，出现00M happens的情况，分为三种情况：

<p>①hyperas调参时，默认返回的是</p>

```python
# return {'loss': -acc, 'status': STATUS_OK, 'model': model}
```

<p>但是当调参次数max_evals设置很大，比如设置100或者200时，容易出现超出显存，此时，只需要将上面的代码修改为以下部分便可解决：</p>

```python
return {'loss': -acc, 'status': STATUS_OK}
```

<p>②在每一次调参时，在代码部分要加上K.clear_session()</p>

```python
def train():
    K.clear_session()
    model()
    return {'loss': -acc, 'status': STATUS_OK}
```

<p>③keras默认的训练函数为fit，fit函数默认一次性加载全部的数据，使用fit_generator代替fit函数，分批加载数据。</p>

```python
def data_generator(data1, data2, targets, batch_size):
	batches = (len(data1) + batch_size - 1) // batch_size
	while (True):
		for i in range(batches):
			X = data1[i * batch_size:(i + 1) * batch_size]
			X_extra = data2[i * batch_size:(i + 1) * batch_size]
			Y = targets[i * batch_size:(i + 1) * batch_size]
			yield [X, X_extra], Y

# 训练的时候用fit_generator
hist=model.fit_generator(generator=data_generator(X_train,X_extra,y_train,batch_size=batch_size),
                         steps_per_epoch=(len(X_train)+batch_size-1)//batch_size,
                         epochs=150,verbose=1,callbacks=callback_list,
                         validation_data=([X_validate,X_validate_extra],y_validate))
```
