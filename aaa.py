#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import tensorflow
print 'keras version',keras.__version__
print 'tensorflow version',tensorflow.__version__

import h5py
import numpy as np
from sklearn.utils import shuffle
np.random.seed(2017) #设置了 numpy 的随机数种子为2017，这样可以确保每个人跑这个代码，输出都能是一样的结果
"""
将多个特征融合到一起，并随机打乱
"""
#
# def feature_load():
#     X_train = []
#     X_test = []
#     for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
#         with h5py.File(filename, 'r') as h:
#             X_train.append(np.array(h['train']))
#             X_test.append(np.array(h['test']))
#             y_train = np.array(h['label'])
#
#     X_train = np.concatenate(X_train, axis=1)
#     X_test = np.concatenate(X_test, axis=1)
#
#     X_train, y_train = shuffle(X_train, y_train)
#     return X_train,y_train,X_test

# X_train = []
# X_test = []
# # for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
# with h5py.File("gap2_ResNet50.h5", 'r') as h:
#     # X_train.append(np.array(h['train']))
#     # X_test.append(np.array(h['test']))
#     # y_train = np.array(h['label'])
#     print h['train']
# from keras.models import load_model
# from keras.utils.visualize_util import plot
#
# model = load_model('my_model.h5')
#
#
# plot(model, to_file='model.png',show_shapes=True)