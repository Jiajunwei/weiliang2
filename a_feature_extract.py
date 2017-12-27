#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py
"""
 采用预训练模型ResNet50、InceptionV3、Xception等，得到训练集、测试集的特征并保存到h5文件中
"""
#训练集（验证集从训练集中分）
TRAIN_PATH = "../pigdata/train"
#测试集
TEST_PATH = "../pigdata/test"

def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(TRAIN_PATH, image_size, shuffle=False,
                                              batch_size=16)
    #测试集可以先不加载
    test_generator = gen.flow_from_directory(TEST_PATH, image_size, shuffle=False,
                                             batch_size=16, class_mode=None)

    train = model.predict_generator(train_generator, train_generator.nb_sample)
    test = model.predict_generator(test_generator, test_generator.nb_sample)
    with h5py.File("gap2_%s.h5" % MODEL.__name__) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

if __name__ == '__main__':
    # write_gap(ResNet50, (224, 224))
    write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
    #write_gap(Xception, (299, 299), xception.preprocess_input)

    # write_gap(VGG16, (224, 224))
    # write_gap(VGG19, (224, 224))

