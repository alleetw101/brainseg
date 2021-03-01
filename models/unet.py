import tensorflow as tf
from tensorflow.keras import layers


def downsample_block(inputs, filters, size):
    x = layers.Conv2D(filters, size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    y = layers.ReLU()(x)

    x = layers.MaxPool2D()(y)

    return x, y


def upsample_block(inputs, inputs_skip, filters, size):
    x = layers.Conv2DTranspose(filters, size, strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, inputs_skip])
    x = layers.Conv2D(filters, size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def bottom_block(inputs, filters, size):
    x = layers.Conv2D(filters, size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x, skip1 = downsample_block(inputs, 64, 3)
    x, skip2 = downsample_block(x, 128, 3)
    x, skip3 = downsample_block(x, 256, 3)
    x, skip4 = downsample_block(x, 512, 3)

    x = bottom_block(x, 1024, 3)

    x = upsample_block(x, skip4, 512, 3)
    x = upsample_block(x, skip3, 256, 3)
    x = upsample_block(x, skip2, 128, 3)
    x = upsample_block(x, skip1, 64, 3)

    output = layers.Conv2D(1, 3, padding='same')(x)

    out_model = tf.keras.Model(inputs, output, name='Unet')

    return out_model


model = unet_model((192, 256, 1))

model.summary()
