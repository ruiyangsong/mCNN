# mCNN 
**Predicting Protein Stability Changes upon Single Point Mutation**

## Directory Structure  
$ ls -F   
README.md  dataset/  model/  result/  shell/  src/
## issues
### FailedPreconditionError: lack of intialization of Keras variables when using with TensorFlow
https://github.com/keras-team/keras/issues/5427  
I initialize the variables with the following code, and its work for me:
```python
K.set_session(tf.Session(graph=model.output.graph))
init = K.tf.global_variables_initializer()
K.get_session().run(init)
```
where K is from 'from keras import backend as K'. tf is from 'import tensorflow as tf'. And 'model' is my keras model. I add this code after compile the model.

### Some trics about keras, see the following markdown file.
[keras+hyperas+multi-GPU.md](keras+hyperas+multi-GPU.md)

## reference
* R Song, H Meng. mCNN: A Convolutional Neural Network Based Method for Predicting Protein
Stability Changes upon Single Point Mutation., 2019.