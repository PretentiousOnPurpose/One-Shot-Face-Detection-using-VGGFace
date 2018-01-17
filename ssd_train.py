import glob
import numpy as np
from ssd_model import SSD
from PIL import Image
# from bbox_utils import default_boxes
from keras.models import Sequential
from keras.layers import Conv2D
import tensorflow as tf

imgs = np.load("imgs.npy")
labels = np.load("labels.npy")

data = []
for img in imgs:
    data.append(np.array(Image.open(img)))

data = np.array(data)

sess = tf.InteractiveSession()
X = tf.placeholder(dtype=tf.float32, shape=((375, 500, 3)))

model = SSD()
model(X)
p1 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(model.get_layer(index=13).output)
p2 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(model.get_layer(index=19).output)
p3 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(model.get_layer(index=26).output)
p4 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(model.get_layer(index=33).output)

def NMS_S1(p1, p2, p3, p4):
    pred1 = tf.reshape(p1, (-1, 46*62, 5))
    pred2 = tf.reshape(p2, (-1, 46*62, 5))
    pred3 = tf.reshape(p3, (-1, 23*31, 5))
    pred4 = tf.reshape(p4, (-1, 11*15, 5))

    Z1 = tf.image.non_max_suppression(boxes = tf.reshape(pred1[:, :, 1:], ((46*62, 4))), scores= tf.reshape(pred1[:, :, 0], ((46*62, ))), max_output_size=5, iou_threshold=0.55)
    Z2 = tf.image.non_max_suppression(boxes = tf.reshape(pred2[:, :, 1:], ((46*62, 4))), scores= tf.reshape(pred2[:, :, 0], ((46*62, ))), max_output_size=5, iou_threshold=0.55)
    Z3 = tf.image.non_max_suppression(boxes = tf.reshape(pred3[:, :, 1:], ((23*31, 4))), scores= tf.reshape(pred3[:, :, 0], ((23*31, ))), max_output_size=5, iou_threshold=0.55)
    Z4 = tf.image.non_max_suppression(boxes = tf.reshape(pred4[:, :, 1:], ((11*15, 4))), scores= tf.reshape(pred4[:, :, 0], ((11*15, ))), max_output_size=5, iou_threshold=0.55)

    A1 = np.array([pred1[0][i] for i in sess.run(Z1)])
    A2 = np.array([pred2[0][i] for i in sess.run(Z2)])
    A3 = np.array([pred3[0][i] for i in sess.run(Z3)])
    A4 = np.array([pred4[0][i] for i in sess.run(Z4)])

    return tf.concat(values=[A1, A2, A3, A4], axis=1)

def NMS_S2(nms):

    nms1 = tf.reshape(nms, (-1, 5))

    Z1 = tf.image.non_max_suppression(boxes=nms1[:, 1:], scores=nms1[:, 0], max_output_size=5, iou_threshold=0.6)
    A1 = np.array([nms1[i] for i in Z1])

    return tf.reshape(A1, (-1, 5))


nms = NMS_S1(p1, p2, p3, p4)
Y = NMS_S2(nms)

loss = tf.losses.mean_squared_error(labels= labels, predictions= Y)
opt = tf.train.AdamOptimizer(learning_rate=0.5)
train_ = opt.minimize(loss)

init = tf.global_variables_initializer()

def train(epochs = 50, In = True):
    if In:
        sess.run(init)
    for e in range(epochs):
        for img in data:
            _, l = sess.run([train_, loss], feed_dict={X:img})
        print("epoch {} , Loss {}".format(e, l/4007))

