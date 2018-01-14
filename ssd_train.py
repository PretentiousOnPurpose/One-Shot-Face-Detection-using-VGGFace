import glob
import numpy as np
from ssd_model import SSD
from bbox_utils import default_boxes
from keras.models import Sequential
from keras.layers import Conv2D

imgs = np.load("imgs.npy")
labels = np.load("labels.npy")

model = SSD()

s0 = Sequential()
s0.add(model.get_layer(index=13))
s0.add(Conv2D(filters=5, kernel_size=3, strides=1, padding='same'))

s1 = Sequential()
s1.add(model.get_layer(index=19))
s1.add(Conv2D(filters=5, kernel_size=3, strides=1, padding='same'))

s2 = Sequential()
s2.add(model.get_layer(index=26))
s2.add(Conv2D(filters=5, kernel_size=3, strides=1, padding='same'))

s3 = Sequential()
s3.add(model.get_layer(index=33))
s3.add(Conv2D(filters=5, kernel_size=3, strides=1, padding='same'))

s4 = Sequential()
s4.add(model.get_layer(index=40))
s4.add(Conv2D(filters=5, kernel_size=3, strides=1, padding='same'))

s5 = Sequential()
s5.add(model.get_layer(index=42))
s5.add(Conv2D(filters=5, kernel_size=3, strides=1, padding='same'))

def NMS_S1():
    pass

def NMS_S2():
    pass

def train(epochs = 50):
    pass

print(imgs[1])
print(labels[1])