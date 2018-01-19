import glob
import numpy as np
from ssd_model import SSD
from PIL import Image
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D

imgs = np.load("imgs.npy")
labels = np.load("labels.npy")

data = []
for img in imgs:
    data.append(np.array(Image.open(img)))

data = np.array(data)

sess = K.get_session()
X = K.placeholder(shape=((None, 375, 500, 3)))
Y = K.placeholder(shape=((None, -1, 5)))
model = SSD()
model(X)
p1 = Conv2D(filters=5, kernel_size=7, strides=1, padding='same')(model.get_layer(index=13).output)
p2 = Conv2D(filters=5, kernel_size=5, strides=1, padding='same')(model.get_layer(index=19).output)
p3 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(model.get_layer(index=26).output)
p4 = Conv2D(filters=5, kernel_size=1, strides=1, padding='same')(model.get_layer(index=33).output)

def NMS(p1, p2, p3, p4):
    pred1 = K.reshape(p1, (46 * 62, 5))
    pred2 = K.reshape(p2, (46 * 62, 5))
    pred3 = K.reshape(p3, (23 * 31, 5))
    pred4 = K.reshape(p4, (11 * 15, 5))

    Z1 = K.tf.image.non_max_suppression(boxes=K.reshape(pred1[:, 1:], ((46 * 62, 4))),
                                        scores=K.reshape(pred1[:, 0], ((46 * 62,))), max_output_size=10,
                                        iou_threshold=0.55)
    Z2 = K.tf.image.non_max_suppression(boxes=K.reshape(pred2[:, 1:], ((46 * 62, 4))),
                                        scores=K.reshape(pred2[:, 0], ((46 * 62,))), max_output_size=10,
                                        iou_threshold=0.55)
    Z3 = K.tf.image.non_max_suppression(boxes=K.reshape(pred3[:, 1:], ((23 * 31, 4))),
                                        scores=K.reshape(pred3[:, 0], ((23 * 31,))), max_output_size=10,
                                        iou_threshold=0.55)
    Z4 = K.tf.image.non_max_suppression(boxes=K.reshape(pred4[:, 1:], ((11 * 15, 4))),
                                        scores=K.reshape(pred4[:, 0], ((11 * 15,))), max_output_size=10,
                                        iou_threshold=0.55)

    A1 = K.gather(pred1, Z1)
    A2 = K.gather(pred2, Z2)
    A3 = K.gather(pred3, Z3)
    A4 = K.gather(pred4, Z4)

    A = K.concatenate(tensors=[K.gather(pred1, Z1), K.gather(pred2, Z2), K.gather(pred3, Z3), K.gather(pred4, Z4)],
                      axis=0)

    Z5 = K.tf.image.non_max_suppression(boxes=K.reshape(A[:, 1:], (-1, 4)), scores=K.reshape(A[:, 0], (-1,)),
                                        max_output_size=10, iou_threshold=0.6)
    return K.gather(A, Z5)

def loss(Y, y_):
    _, D, _ = Y.get_shape().as_list()
    if D > 1:
        pass
    elif D == 1:
        pass

y_ = NMS(p1, p2, p3, p4)
Loss = loss(Y, y_)

opt = K.tf.train.AdamOptimizer(learning_rate=0.05)
train_ = opt.minimize(Loss)

init = K.tf.global_variables_initializer()

def train(epochs = 100):
    for e in range(epochs):
        _, L = sess.run([train_, Loss], feed_dict={X: data, Y: labels})
        if e % 10 == 0:
            print("epoch {} , loss {}".format(e, L))
