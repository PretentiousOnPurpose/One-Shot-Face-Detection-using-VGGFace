from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, AveragePooling2D

def SSD():
    vgg = VGGFace(include_top=False, input_shape=(300, 400, 3))
    for layer in vgg.layers:
        layer.trainable = False

    ssd = Model(inputs=vgg.inputs, outputs=vgg.get_layer(index=13).output)
    model = Sequential()
    model.add(ssd)

    model.add(Conv2D(filters=1024, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=1024, kernel_size=1, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=256, kernel_size=1, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, kernel_size=1, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, kernel_size=1, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(AveragePooling2D(strides=2))

    return model