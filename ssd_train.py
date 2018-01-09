import os
import glob
import keras
import mxnet as mx
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from ssd_model import ssd, class_pred, box_pred
from mxnet.contrib.ndarray import MultiBoxDetection
from bbox_utils import default_boxes, box_to_rect, plot_boxes

EPOCHS = 40

SSD = ssd(300)

imgs = glob.glob("data/*.jpg")
data = np.array([np.array(Image.open(img)) for img in imgs])

for e in range(EPOCHS):
    for img in data:
        s0 = Model(inputs=SSD.inputs, outputs=SSD.get_layer(index=0).output)(img)
        s1 = Model(inputs=SSD.inputs, outputs=SSD.get_layer(index=7).output)(img)
        s2 = Model(inputs=SSD.inputs, outputs=SSD.get_layer(index=14).output)(img)
        s3 = Model(inputs=SSD.inputs, outputs=SSD.get_layer(index=21).output)(img)
        s4 = Model(inputs=SSD.inputs, outputs=SSD.get_layer(index=28).output)(img)
        s5 = Model(inputs=SSD.inputs, outputs=SSD.get_layer(index=35).output)(img)
        s6 = SSD.predict(img)

        boxes = box_pred

        

