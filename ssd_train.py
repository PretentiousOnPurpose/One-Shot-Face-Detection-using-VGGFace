import keras
import numpy as np
from PIL import Image
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv2D
from misc import load_data
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D

data = load_data()

K.set_learning_phase(1)

sess = K.get_session()
X = K.tf.placeholder(shape=(None, 375, 500, 3), dtype=K.tf.float32)
Y = K.tf.placeholder(shape=(None, 5), dtype=K.tf.float32)

vgg = VGGFace(include_top=False, input_tensor = X, input_shape=(375, 500, 3))
for layer in vgg.layers:
    layer.trainable = False

Z1 = (Conv2D(filters=1024, kernel_size=3, strides=1, padding='same'))(vgg.get_layer(index=13).output)
Z2 = (BatchNormalization())(Z1)
Z3 = (Activation('relu'))(Z2)
Z4 = (Conv2D(filters=1024, kernel_size=1, strides=1, padding='same'))(Z3)
Z5 = (BatchNormalization())(Z4)
Z6 = (Activation('relu'))(Z5)
Z7 = (MaxPool2D(pool_size=2, strides=2))(Z6)

Z8 = (Conv2D(filters=256, kernel_size=1, strides=1, padding='same'))(Z7)
Z9 = (BatchNormalization())(Z8)
Z10 = (Activation('relu'))(Z9)
Z11 = (Conv2D(filters=512, kernel_size=3, strides=1, padding='same'))(Z10)
Z12 = (BatchNormalization())(Z11)
Z13 = (Activation('relu'))(Z12)
Z14 = (MaxPool2D(pool_size=2, strides=2))(Z13)

# Z15 = (Conv2D(filters=128, kernel_size=1, strides=1, padding='same'))(Z14)
# Z16 = (BatchNormalization())(Z15)
# Z17 = (Activation('relu'))(Z16)
# Z18 = (Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))(Z17)
# Z19 = (BatchNormalization())(Z18)
# Z20 = (Activation('relu'))(Z19)
# Z21 = (MaxPool2D(pool_size=2, strides=2))(Z20)


p1 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(vgg.get_layer(index=13).output)
p2 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(Z7)
p3 = Conv2D(filters=5, kernel_size=3, strides=1, padding='same')(Z14)
p4 = Conv2D(filters=5, kernel_size=1, strides=1, padding='same')(Z14)


def NMS(p1, p2, p3, p4):
    pred1 = K.reshape(p1, (46 * 62, 5))
    pred2 = K.reshape(p2, (23 * 31, 5))
    pred3 = K.reshape(p3, (11 * 15, 5))
    pred4 = K.reshape(p4, (11 * 15, 5))

    Z1 = K.tf.image.non_max_suppression(boxes=K.reshape(pred1[:, 1:], ((46 * 62, 4))),
                                        scores=K.reshape(pred1[:, 0], ((46 * 62,))), max_output_size=10,
                                        iou_threshold=0.55)
    Z2 = K.tf.image.non_max_suppression(boxes=K.reshape(pred2[:, 1:], ((23 * 31, 4))),
                                        scores=K.reshape(pred2[:, 0], ((23 * 31,))), max_output_size=10,
                                        iou_threshold=0.55)
    Z3 = K.tf.image.non_max_suppression(boxes=K.reshape(pred3[:, 1:], ((11* 15, 4))),
                                        scores=K.reshape(pred3[:, 0], ((11 * 15,))), max_output_size=10,
                                        iou_threshold=0.55)
    Z4 = K.tf.image.non_max_suppression(boxes=K.reshape(pred4[:, 1:], ((11 * 15, 4))),
                                        scores=K.reshape(pred4[:, 0], ((11 * 15,))), max_output_size=10,
                                        iou_threshold=0.55)

    A = K.concatenate(tensors=[K.gather(pred1, Z1), K.gather(pred2, Z2), K.gather(pred3, Z3), K.gather(pred4, Z4)],
                      axis=0)

    Z5 = K.tf.image.non_max_suppression(boxes=K.reshape(A[:, 1:], (-1, 4)), scores=K.reshape(A[:, 0], (-1,)),
                                        max_output_size=10, iou_threshold=0.6)
    return K.gather(A, Z5)


def sigmoid(X):
    return 1/(1 + K.exp(-X))

def relu(X):
    return K.maximum(X, 0)

def loss(Y, y_):
    print(Y.shape.as_list())
    print(y_.shape.as_list())
    if Y.shape.as_list() != y_.shape.as_list():
        print("Clicked")
        Y = K.zeros_like(y_)
    # softMax = -(Y[:, 0] * K.log(sigmoid(y_[:, 0])) + ((1 - Y[:, 0]) * K.log(sigmoid(1 - y_[:, 0]))))
    # mse = keras.losses.mean_squared_error(Y[:, 1:], relu(y_[:, 1:]))
    # return softMax + mse
    return 1.0
y_ = NMS(p1, p2, p3, p4)

Loss = loss(Y, y_)
# opt = K.tf.train.AdamOptimizer(learning_rate=0.05)
# train_ = opt.minimize(Loss)
sess.run(K.tf.global_variables_initializer())
h = sess.run(y_, feed_dict={X: np.array(Image.open("x.jpg")).reshape((1, 375, 500, 3))})
print(h.shape)


def train(epochs = 100):
    for e in range(epochs):
        for i, d in enumerate(data):
            _, L = sess.run([train_, Loss], feed_dict={X: np.array(Image.open("x.jpg")).reshape((1, 375, 500, 3)), Y: d[1].reshape((-1, 5))})
            if e % 10 == 0:
                print("epoch {} , loss {}".format(e, L))

# train(epochs = 1)