import tensorflow as tf
import numpy as np
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D

NUM_Class = 1+1 # One for face and other for Background
NUM_Anchors = 5

def ssd(input_dims):
    vgg_base = keras.applications.VGG16(include_top=False, input_shape=(input_dims, input_dims, 3), classes=2)
    model = Model(inputs=vgg_base.input, outputs=vgg_base.get_layer(index=13).output)
    ssd = Sequential()
    ssd.add(model)
    ssd.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(MaxPooling2D(2))
    # Scale 1 Prediction
    ssd.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(Conv2D(filters=1024, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(MaxPooling2D(1))
    # Scale 2 Prediction
    ssd.add(Conv2D(filters=512, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(Conv2D(filters=512, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(MaxPooling2D(2))
    # Scale 3 Prediction
    ssd.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(MaxPooling2D(2))
    # Scale 4 Prediction
    ssd.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(MaxPooling2D(2))
    # Scale 5 Prediction
    ssd.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    ssd.add(BatchNormalization())
    ssd.add(Activation('relu'))
    ssd.add(MaxPooling2D(2))

    return ssd

def class_pred(feat):
    return Conv2D(filters=10, kernel_size=3, padding='same')

def box_pred(feat):
    return Conv2D(filters=20, kernel_size=3, padding='same')

