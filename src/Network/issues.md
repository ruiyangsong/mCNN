## FailedPreconditionError: lack of intialization of Keras variables when using with TensorFlow
https://github.com/keras-team/keras/issues/5427  
I initialize the variables with the following code, and its work for me:
```python
K.set_session(tf.Session(graph=model.output.graph))
init = K.tf.global_variables_initializer()
K.get_session().run(init)
```
where K is from 'from keras import backend as K'. tf is from 'import tensorflow as tf'. And 'model' is my keras model. I add this code after compile the model.

## [keras+hyperas+multi-GPU](keras+hyperas+multi-GPU.md)