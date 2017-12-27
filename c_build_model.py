#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 0020 上午 11:12
# @Author  : Lovin
from keras.models import *
from keras.layers import *
from b_feature_load import feature_load
import h5py
from keras.utils import np_utils
n_class = 30



def build_model():
    inputs = Input(X_train.shape[1:])
    x = inputs
    x = Dropout(0.5)(x)
    x = Dense(30, activation='softmax')(x)
    model = Model(inputs, x)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # 加载特征数据
    print '加载特征数据...'
    X_train, y_train, _ = feature_load()
    # print X_train.shape
    # print X_train.shape[1:]

    # print y_train.shape
    y_train = np_utils.to_categorical(y_train, n_class)
    # print y_train.shape

    model = build_model()
    print '训练...'

    history = model.fit(X_train, y_train, batch_size=64, nb_epoch=8, validation_split=0.2)
    # print history
    # print '保存模型...'
    model.save('my_model.h5')


