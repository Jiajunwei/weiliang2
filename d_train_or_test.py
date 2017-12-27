#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/20 0020 下午 2:28
# @Author  : Lovin
import pandas as pd
from keras.preprocessing.image import *
from keras.utils import np_utils
from keras.models import load_model
from c_build_model import build_model
from b_feature_load import feature_load
import math


TEST_PATH= '../pigdata/test'
TRAIN = False
n_class = 30

X_train, y_train, X_test = feature_load()
y_train = np_utils.to_categorical(y_train, n_class)


if TRAIN:
    model = build_model()
    model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2)
    model.save('my_model.h5')
else:
    model = load_model('my_model.h5')
    y_pred = model.predict(X_test, verbose=1)
    #设置概率的最大最小值
    y_pred = y_pred.clip(min=math.pow(10,-15), max=1-math.pow(10,-15))
    # y_pred = y_pred.clip(min=0.005, max=0.995)
    #结果样例
    df = pd.read_csv("sample_submission.csv")

    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(TEST_PATH, (224, 224), shuffle=False,
                                             batch_size=16, class_mode=None)


    for i, fname in enumerate(test_generator.filenames):
        # print i,fname
        img_name = int(fname[fname.rfind('/') + 1:fname.rfind('.')])
        for j in range(len(y_pred[0])):
            # df.set_value(img_name, j+1, y_pred[i][j])
            df.iat[i*30 + j, 0] = img_name
            df.iat[i*30 + j, 1] = j+1
            df.iat[i*30 + j, 2] = y_pred[i][j]
            # df.iat[i * 30 + j, :] = [img_name,j+1,y_pred[i][j]]

    df.to_csv('pred2.csv', index=None)

