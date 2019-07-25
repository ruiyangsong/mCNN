#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file_name : test.py
# time      : 3/11/2019 13:00
# author    : ruiyang
# email     : ww_sry@163.com


import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import cdist
from Bio.PDB.PDBParser import PDBParser

from sklearn.datasets import make_classification
from collections import Counter


from keras.models import Model
from keras import layers
from keras import Input

# X, y = make_classification(n_samples=5000, n_features=3, n_informative=2,
#                            n_redundant=0, n_repeated=0, n_classes=3,
#                            n_clusters_per_class=1,
#                            weights=[0.01, 0.05, 0.94],
#                            class_sep=0.8, random_state=0)
# print(X.shape,y.shape)

# ## ========== test VGG-like convnet ========== ##
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.optimizers import SGD
#
# # Generate dummy data
# x_train = np.random.random((100, 100, 100, 3))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# x_test = np.random.random((20, 100, 100, 3))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
#
# model = Sequential()
# # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# # this applies 32 convolution filters of size 3x3 each.
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
# model.fit(x_train, y_train, batch_size=32, epochs=10)
# score = model.evaluate(x_test, y_test, batch_size=32)
# print('score: ',score)







## ========== test plt ========== ##
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# x = np.arange(100)
# y = np.sin(x)
# plt.plot(x,y)
# plt.show()


## ========== test PCA ========== ##

from sklearn.decomposition import PCA

# array = np.array([[0,0,0],[0,2,0],[0,2,1]])
# array=np.array([[2,0,0],[0,0,0],[0,0,1]])
# array = np.array([[1,0,0],[0,3**(1/2),0],[0,3**(1/2),1]])
# array = np.array([[0,0],[1,1]])
# print('原空间坐标：\n',array)
# pca = PCA(n_components = 2)
# # Y = pca.fit_transform(array)
#
# pca.fit(array)
# print('正交基为: \n',pca.components_)
#
# X = pca.transform(array)
# print('转换后：\n',X)

# X1 = np.array([np.dot(array[0],pca.components_[0]),np.dot(array[0],pca.components_[1]),np.dot(array[0],pca.components_[2])])
# X2 = np.array([np.dot(array[1],pca.components_[0]),np.dot(array[1],pca.components_[1]),np.dot(array[1],pca.components_[2])])
# X3 = np.array([np.dot(array[2],pca.components_[0]),np.dot(array[2],pca.components_[1]),np.dot(array[2],pca.components_[2])])
# # print('人为验证：\n',X1,X2,X3)
# print('人为验证2：\n',np.dot(array,np.transpose(pca.components_)))
# print(array[1].shape)
# Y1 = pca.transform(array[1].reshape(-1,3)) #旋转之后的坐标数组
# print('转换后的Y:\n',Y1)