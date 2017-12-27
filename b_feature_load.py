#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 0020 上午 11:08
# @Author  : Lovin
# @Site    : 
# @File    : load_feature.py
# @Software: PyCharm

import h5py
import numpy as np
from sklearn.utils import shuffle
np.random.seed(2017) #设置了 numpy 的随机数种子为2017，这样可以确保每个人跑这个代码，输出都能是一样的结果
"""
将多个特征融合到一起，并随机打乱
"""

def feature_load():
    X_train = []
    X_test = []
    for filename in ["gap2_ResNet50.h5", "gap2_Xception.h5", "gap2_InceptionV3.h5"]:
    # for filename in ["gap2_ResNet50.h5", "gap2_Xception.h5"]:
        with h5py.File(filename, 'r') as h:
            X_train.append(np.array(h['train']))
            X_test.append(np.array(h['test']))
            y_train = np.array(h['label'])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    X_train, y_train = shuffle(X_train, y_train)
    return X_train,y_train,X_test
