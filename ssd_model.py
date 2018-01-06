import tensorflow as tf
import numpy as np
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D

NUM_Class = 1+1 # One for face and other for Background
NUM_Anchors = 5

def downSample(feat, filters, pool):
    seq = Sequential()
    for i in range(2):
        seq.add(Conv2D(filters=filters, kernel_size=3, padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=pool))
    return seq

def ssd(input_dims):
    vgg_base = keras.applications.VGG16(include_top=False, input_shape=(input_dims, input_dims, 3), classes=2)
    model = Model(inputs=vgg_base.input, outputs=vgg_base.get_layer(index=13).output)
    ssd = Sequential()
    ssd.add(model)
    ssd.add(downSample(ssd.get_layer(index=13).output, 1024, 2))
    # Scale 1 Prediction
    sdd.add(downSample(ssd.get_layer(index=14).output, 1024, 1))
    # Scale 2 Prediction
    sdd.add(downSample(ssd.get_layer(index=15).output, 512, 2))
    # Scale 3 Prediction
    sdd.add(downSample(ssd.get_layer(index=16).output, 256, 2))
    # Scale 4 Prediction
    sdd.add(downSample(ssd.get_layer(index=17).output, 256, 2))
    # Scale 5 Prediction
    sdd.add(downSample(ssd.get_layer(index=17).output, 256, 2))    

    return ssd

def class_pred(feat):
    return Conv2D(filters=10, kernel_size=3, padding='same')

def box_pred(feat):
    return Conv2D(filters=20, kernel_size=3, padding='same')

